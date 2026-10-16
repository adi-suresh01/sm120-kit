use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};

use crate::cloud::PodStatus;
use crate::ledger::PodEntry;
use crate::pod::{Ctx, POLL};

pub const KIT_DIR: &str = "/workspace/sm120-kit";
const UPLOAD_EXCLUDES: [&str; 3] = [".DS_Store", "harness-ledger.json*", ".harness-ledger.lock"];
/// A poll, a job start, or the topology gate answers in seconds on a healthy pod. ServerAlive ends a dead
/// connection, but a remote command that hangs (nvidia-smi on a wedged GPU) holds ssh open for good, and every
/// loop that bills waits on these calls.
pub const COMMAND_LIMIT: Duration = Duration::from_secs(120);
/// Copying the kit up or results down: tens of MB over whatever link the host has.
pub const TRANSFER_LIMIT: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Target {
    pub host: String,
    pub port: u16,
    pub known_hosts: PathBuf,
}

/// Everything the harness does on a pod. `SshRemote` is the real one; tests use a fake.
pub trait Remote {
    /// Fails before any money is spent if ssh could not log in without a prompt.
    fn preflight(&self) -> Result<()>;
    fn reachable(&self, target: &Target) -> bool;
    /// Runs a bash script in the kit directory with the container environment loaded and returns its stdout.
    fn run(&self, target: &Target, script: &str) -> Result<String>;
    fn run_streaming(&self, target: &Target, script: &str) -> Result<()>;
    /// `path` is relative to the kit directory.
    fn write_file(&self, target: &Target, path: &str, content: &str) -> Result<()>;
    fn upload(&self, target: &Target, local_root: &Path, paths: &[&str]) -> Result<()>;
    /// `paths` are relative to the kit directory; the ones that do not exist on the pod are skipped. Gives up
    /// after `limit`.
    fn download(&self, target: &Target, paths: &[&str], local_root: &Path, limit: Duration) -> Result<()>;
}

pub fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

/// Images that run Python from a venv set through ENV (lmsysorg/sglang) do not pass it to SSH sessions, so every
/// script loads the environment the entrypoint saved.
pub fn remote_script(script: &str) -> String {
    format!("set -euo pipefail\nsource /root/container_env.sh\nmkdir -p {KIT_DIR}\ncd {KIT_DIR}\n{script}\n")
}

pub fn ssh_banner(host: &str, port: u16, timeout: Duration) -> bool {
    let Some(addr) = (host, port).to_socket_addrs().ok().and_then(|mut addrs| addrs.next()) else {
        return false;
    };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, timeout) else {
        return false;
    };
    let mut banner = [0u8; 4];
    stream.set_read_timeout(Some(timeout)).is_ok() && stream.read_exact(&mut banner).is_ok() && &banner == b"SSH-"
}

/// The real remote, over the system OpenSSH.
pub struct SshRemote {
    identity: Option<PathBuf>,
}

impl SshRemote {
    pub fn new(identity: Option<PathBuf>) -> Self {
        Self { identity }
    }

    /// HARNESS_SSH_KEY names the key registered with RunPod, so ssh offers only that one: an agent holding many
    /// keys can use up the pod's MaxAuthTries before it reaches the right one.
    pub fn from_env() -> Self {
        Self::new(
            std::env::var_os("HARNESS_SSH_KEY")
                .filter(|key| !key.is_empty())
                .map(PathBuf::from),
        )
    }

    fn ssh(&self, target: &Target) -> Result<Command> {
        ssh_command(target, self.identity.as_deref())
    }

    fn run_script(&self, target: &Target, script: &str, output: Output, limit: Duration) -> Result<Vec<u8>> {
        let mut ssh = self.ssh(target)?;
        ssh.arg("bash -s");
        let what = format!("remote script on {}:{}", target.host, target.port);
        run_bounded(ssh, remote_script(script).as_bytes(), output, limit, &what)
    }
}

fn ssh_command(target: &Target, identity: Option<&Path>) -> Result<Command> {
    if let Some(dir) = target.known_hosts.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut command = Command::new("ssh");
    // The pod runs a third-party image as root; a forwarded agent would let it log in anywhere as Aditya, and a
    // SendEnv pattern in a user's ssh config would hand it the API key.
    command
        .env_remove("RUNPOD_API_KEY")
        .args([
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ForwardAgent=no",
        ])
        .arg("-o")
        .arg(format!("UserKnownHostsFile=\"{}\"", target.known_hosts.display()));
    if let Some(identity) = identity {
        command
            .args(["-o", "IdentitiesOnly=yes", "-o"])
            .arg(format!("IdentityFile=\"{}\"", identity.display()));
    }
    command
        .args([
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=4",
        ])
        .args(["-o", "LogLevel=ERROR", "-p"])
        .arg(target.port.to_string())
        .arg(format!("root@{}", target.host));
    Ok(command)
}

fn quietly_succeeds(command: &mut Command) -> bool {
    command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Waits for every process of a pipeline and kills all of them once `limit` passes.
fn wait_within(children: &mut [&mut Child], limit: Duration, what: &str) -> Result<Vec<ExitStatus>> {
    let started = Instant::now();
    let mut statuses: Vec<Option<ExitStatus>> = vec![None; children.len()];
    loop {
        for (child, status) in children.iter_mut().zip(statuses.iter_mut()) {
            if status.is_none() {
                *status = child.try_wait().with_context(|| format!("waiting for {what}"))?;
            }
        }
        if statuses.iter().all(Option::is_some) {
            return Ok(statuses.into_iter().flatten().collect());
        }
        if started.elapsed() >= limit {
            for child in children.iter_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            bail!("{what} ran past its {limit:?} limit and was killed");
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[derive(Clone, Copy)]
enum Output {
    Capture,
    Inherit,
    Discard,
}

/// Runs `command` with `input` on stdin, bounded by `limit`. Stdin and stdout are pumped from their own threads,
/// so a peer that stops reading or writing can never block the wait for the limit.
fn run_bounded(mut command: Command, input: &[u8], output: Output, limit: Duration, what: &str) -> Result<Vec<u8>> {
    let stdout = match output {
        Output::Capture => Stdio::piped(),
        Output::Inherit => Stdio::inherit(),
        Output::Discard => Stdio::null(),
    };
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(stdout)
        .spawn()
        .with_context(|| format!("starting {what}"))?;
    let mut stdin = child.stdin.take().context("child stdin")?;
    let input = input.to_vec();
    let writer = thread::spawn(move || stdin.write_all(&input));
    let reader = child.stdout.take().map(|mut out| {
        thread::spawn(move || {
            let mut captured = Vec::new();
            out.read_to_end(&mut captured).map(|_| captured)
        })
    });
    let status = wait_within(&mut [&mut child], limit, what)?[0];
    let written = writer.join().map_err(|_| anyhow!("{what}: stdin writer panicked"))?;
    let captured = match reader {
        Some(reader) => reader
            .join()
            .map_err(|_| anyhow!("{what}: stdout reader panicked"))?
            .with_context(|| format!("reading the output of {what}"))?,
        None => Vec::new(),
    };
    ensure!(status.success(), "{what} failed ({status})");
    // A script cut short on stdin would have run only its first lines, so a failed write is an error even when
    // the remote side exited 0.
    written.with_context(|| format!("writing the input of {what}"))?;
    Ok(captured)
}

/// The remote side of a download, as POSIX sh because the pod's login shell runs it. GNU tar exits 1 when a file
/// changed while it was read (a server log still being written), and the archive is still whole.
fn download_command(paths: &[&str]) -> String {
    let quoted: Vec<String> = paths.iter().map(|path| shell_quote(path)).collect();
    format!(
        "cd {KIT_DIR} || exit 2; set --; for p in {}; do if [ -e \"$p\" ]; then set -- \"$@\" \"$p\"; fi; done; \
         tar --warning=no-file-changed -czf - -T /dev/null \"$@\"; rc=$?; [ \"$rc\" -le 1 ]",
        quoted.join(" ")
    )
}

impl Remote for SshRemote {
    /// ssh runs with BatchMode and cannot ask for a passphrase, so it needs a key file without one or a key in
    /// ssh-agent.
    fn preflight(&self) -> Result<()> {
        if let Some(identity) = &self.identity {
            ensure!(
                identity.is_file(),
                "HARNESS_SSH_KEY names {}, which is not a file",
                identity.display()
            );
            let unencrypted = quietly_succeeds(Command::new("ssh-keygen").args(["-y", "-P", "", "-f"]).arg(identity));
            if unencrypted {
                return Ok(());
            }
        }
        ensure!(
            quietly_succeeds(Command::new("ssh-add").arg("-l")),
            "ssh-agent holds no key (ssh-add -l failed); run ssh-add first: ssh runs with BatchMode and cannot ask \
             for a passphrase"
        );
        Ok(())
    }

    fn reachable(&self, target: &Target) -> bool {
        ssh_banner(&target.host, target.port, Duration::from_secs(5))
    }

    fn run(&self, target: &Target, script: &str) -> Result<String> {
        let stdout = self.run_script(target, script, Output::Capture, COMMAND_LIMIT)?;
        Ok(String::from_utf8_lossy(&stdout).into_owned())
    }

    /// `harness ssh` is for a person at the terminal, who can Ctrl-C; the transfer limit still stops one left
    /// running in a background shell.
    fn run_streaming(&self, target: &Target, script: &str) -> Result<()> {
        self.run_script(target, script, Output::Inherit, TRANSFER_LIMIT)
            .map(drop)
    }

    fn write_file(&self, target: &Target, path: &str, content: &str) -> Result<()> {
        let full = format!("{KIT_DIR}/{path}");
        let dir = full.rsplit_once('/').map_or(KIT_DIR, |(dir, _)| dir);
        let mut ssh = self.ssh(target)?;
        ssh.arg(format!("mkdir -p {} && cat > {}", shell_quote(dir), shell_quote(&full)));
        let what = format!("writing {full} on the pod");
        run_bounded(ssh, content.as_bytes(), Output::Discard, COMMAND_LIMIT, &what).map(drop)
    }

    fn upload(&self, target: &Target, local_root: &Path, paths: &[&str]) -> Result<()> {
        let mut tar = Command::new("tar");
        // COPYFILE_DISABLE stops macOS tar from adding AppleDouble ._ files next to every file.
        tar.env("COPYFILE_DISABLE", "1").args(["-czf", "-", "--no-xattrs"]);
        for pattern in UPLOAD_EXCLUDES {
            tar.args(["--exclude", pattern]);
        }
        let mut tar = tar
            .arg("-C")
            .arg(local_root)
            .args(paths)
            .stdout(Stdio::piped())
            .spawn()
            .context("starting tar")?;
        let archive = tar.stdout.take().context("tar stdout")?;
        let mut ssh = self
            .ssh(target)?
            .arg(format!("mkdir -p {KIT_DIR} && tar --no-same-owner -xzf - -C {KIT_DIR}"))
            .stdin(Stdio::from(archive))
            .spawn()
            .context("starting ssh")?;
        let what = format!("copying {paths:?} to the pod");
        let statuses = wait_within(&mut [&mut tar, &mut ssh], TRANSFER_LIMIT, &what)?;
        ensure!(
            statuses.iter().all(ExitStatus::success),
            "{what} failed (tar {}, ssh {})",
            statuses[0],
            statuses[1]
        );
        Ok(())
    }

    fn download(&self, target: &Target, paths: &[&str], local_root: &Path, limit: Duration) -> Result<()> {
        let mut ssh = self
            .ssh(target)?
            .arg(download_command(paths))
            .stdout(Stdio::piped())
            .spawn()
            .context("starting ssh")?;
        let archive = ssh.stdout.take().context("ssh stdout")?;
        fs::create_dir_all(local_root)?;
        let mut tar = Command::new("tar")
            .args(["-xzf", "-", "-C"])
            .arg(local_root)
            .stdin(Stdio::from(archive))
            .spawn()
            .context("starting tar")?;
        let what = format!("pulling {paths:?}");
        let statuses = wait_within(&mut [&mut ssh, &mut tar], limit, &what)?;
        ensure!(
            statuses.iter().all(ExitStatus::success),
            "{what} failed (ssh {}, tar {})",
            statuses[0],
            statuses[1]
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobState {
    Running,
    Exited(i32),
    /// Gone without writing an exit code: killed, or lost with a container restart.
    Dead,
}

#[derive(Debug, PartialEq, Eq)]
pub struct JobStatus {
    pub state: JobState,
    pub lines: Vec<String>,
    pub offset: u64,
}

/// Opens the poll reply, so whatever a login shell prints first (a motd, an echo in .bashrc) is never read as
/// the job's state.
pub const POLL_SENTINEL: &str = "--harness-poll--";

pub fn job_dir(issue: &str) -> String {
    format!("results/{issue}/jobs")
}

/// `alive <pid file>`: the pid is a live process whose command line names that file. A container restart starts
/// the pid namespace over, so a stale pid can belong to sshd or the watchdog, and a zombie still answers kill -0.
/// grep reads all of ps's output, so pipefail never sees ps die of SIGPIPE.
const ALIVE: &str = "alive() {\n\
    \x20 local pid\n\
    \x20 pid=$(cat \"$1\" 2>/dev/null) || return 1\n\
    \x20 kill -0 \"$pid\" 2>/dev/null && ps -o args= -p \"$pid\" 2>/dev/null | grep -F \"$1\" > /dev/null\n\
    }";

/// Starts `command` detached from the SSH session, so a dropped connection never kills a run. The job records its
/// pid first and its exit code by rename last, so a poll never reads a half-written file and can tell a job that
/// died from one that runs. One job per issue at a time: two servers on the same GPUs would spoil both results.
pub fn start_job(remote: &dyn Remote, target: &Target, issue: &str, job: &str, command: &str) -> Result<()> {
    let dir = job_dir(issue);
    let (pid, exit) = (format!("{dir}/{job}.pid"), format!("{dir}/{job}.exit"));
    // The subshell keeps an `exit` in the command from ending the wrapper before it records the code.
    let body = format!(
        "echo $$ > {pid}.tmp && mv {pid}.tmp {pid}\n(\n{command}\n)\necho $? > {exit}.tmp && mv {exit}.tmp {exit}"
    );
    let script = format!(
        "{ALIVE}\n\
         mkdir -p {dir}\n\
         for other in {dir}/*.pid; do\n\
         \x20 [[ -e $other ]] || continue\n\
         \x20 if [[ ! -f ${{other%.pid}}.exit ]] && alive \"$other\"; then\n\
         \x20   name=${{other##*/}}; echo \"BUSY ${{name%.pid}} $(cat \"$other\")\"; exit 0\n\
         \x20 fi\n\
         done\n\
         rm -f {exit} {pid}\n\
         setsid nohup bash -c {} > {dir}/{job}.log 2>&1 < /dev/null &\n\
         for _ in $(seq 50); do\n\
         \x20 if [[ -s {pid} ]]; then echo STARTED; exit 0; fi\n\
         \x20 sleep 0.1\n\
         done",
        shell_quote(&body)
    );
    let output = remote.run(target, &script)?;
    if let Some(busy) = output.lines().find_map(|line| line.strip_prefix("BUSY ")) {
        let (other, pid) = busy.split_once(' ').unwrap_or((busy, "?"));
        bail!(
            "job {other} (pid {pid}) of issue {issue} still runs on the pod, so {job} did not start; wait for it, or \
             end it with `harness ssh {issue} -- kill {pid}`"
        );
    }
    ensure!(
        output.lines().any(|line| line == "STARTED"),
        "job {job} did not start within 5 s; see results/{issue}/jobs/{job}.log on the pod"
    );
    Ok(())
}

/// Reports the job's state and the complete lines added to `tail_path` since `offset`. The state is read first,
/// so once it says the job exited the tail holds everything the job wrote. The exit file is read again after the
/// liveness check, since a job can write it and exit between the two.
pub fn poll_job(
    remote: &dyn Remote,
    target: &Target,
    issue: &str,
    job: &str,
    tail_path: &str,
    offset: u64,
) -> Result<JobStatus> {
    let dir = job_dir(issue);
    let (pid, exit) = (format!("{dir}/{job}.pid"), format!("{dir}/{job}.exit"));
    let script = format!(
        "{ALIVE}\n\
         echo {POLL_SENTINEL}\n\
         if [[ -f {exit} ]]; then echo \"EXIT $(cat {exit})\"\n\
         elif alive {pid}; then echo RUNNING\n\
         elif [[ -f {exit} ]]; then echo \"EXIT $(cat {exit})\"\n\
         else echo DEAD; fi\n\
         tail -c +{} {tail_path} 2>/dev/null || true",
        offset + 1
    );
    parse_poll(&remote.run(target, &script)?, offset)
}

pub fn parse_poll(output: &str, offset: u64) -> Result<JobStatus> {
    let marker = format!("{POLL_SENTINEL}\n");
    let start = if output.starts_with(&marker) {
        0
    } else {
        output
            .find(&format!("\n{marker}"))
            .map(|at| at + 1)
            .with_context(|| format!("poll reply has no {POLL_SENTINEL} line: {:?}", truncate(output)))?
    };
    let reply = &output[start + marker.len()..];
    let (head, rest) = reply.split_once('\n').unwrap_or((reply, ""));
    let state = match head {
        "RUNNING" => JobState::Running,
        "DEAD" => JobState::Dead,
        _ => match head.strip_prefix("EXIT ") {
            Some(code) => JobState::Exited(code.trim().parse().with_context(|| format!("bad exit line {head:?}"))?),
            None => bail!("unexpected poll reply {head:?}"),
        },
    };
    let complete = rest.rfind('\n').map_or("", |end| &rest[..=end]);
    Ok(JobStatus {
        state,
        lines: complete
            .lines()
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
        offset: offset + complete.len() as u64,
    })
}

fn truncate(text: &str) -> String {
    text.chars().take(200).collect()
}

/// About five minutes of failed polls, at POLL apart, before a job counts as lost.
const POLL_FAILURES: u32 = 30;

/// What `wait_job` does for its caller while the job runs.
pub struct Watch<'w> {
    /// Gets every complete line added to the tailed file.
    pub on_line: &'w mut dyn FnMut(&str),
    /// Runs about once a period while the job answers polls, as for a pull of what it wrote so far.
    pub every: Option<(Duration, &'w mut dyn FnMut())>,
}

/// Polls a job started with `start_job` until it exits, handing every complete new line of `tail_path` to the
/// watch. Ends early on Ctrl-C, at the pod deadline, when RunPod reports the pod stopped or gone after a failed poll,
/// or after POLL_FAILURES polls in a row fail.
pub fn wait_job(
    ctx: &Ctx,
    issue: &str,
    entry: &PodEntry,
    target: &Target,
    job: &str,
    tail_path: &str,
    watch: Watch,
) -> Result<i32> {
    let Watch { on_line, mut every } = watch;
    let mut due = every.as_ref().map(|(period, _)| ctx.clock.now() + *period);
    let mut offset = 0;
    let mut failures = 0;
    loop {
        ctx.check_interrupted()?;
        ensure!(
            ctx.clock.now() < entry.deadline,
            "pod {} reached its deadline while {job} ran",
            entry.name
        );
        match poll_job(ctx.remote, target, issue, job, tail_path, offset) {
            Ok(status) => {
                failures = 0;
                status.lines.iter().for_each(|line| on_line(line));
                offset = status.offset;
                match status.state {
                    JobState::Exited(code) => return Ok(code),
                    JobState::Dead => bail!(
                        "job {job} died without an exit code (killed, or the container restarted); see \
                         results/{issue}/jobs/{job}.log"
                    ),
                    JobState::Running => {}
                }
                if let (Some((period, run)), Some(at)) = (every.as_mut(), due)
                    && ctx.clock.now() >= at
                {
                    run();
                    due = Some(ctx.clock.now() + *period);
                }
            }
            Err(e) => {
                failures += 1;
                check_pod_up(ctx, issue, entry, job)?;
                if failures >= POLL_FAILURES {
                    return Err(e).context(format!("lost contact with the pod while {job} ran"));
                }
                ctx.warn(issue, format!("polling {job} failed ({failures} in a row): {e:#}"));
            }
        }
        ctx.clock.sleep(POLL);
    }
}

/// A container RunPod stopped (as it does at about 98% of the container's memory limit) never answers again, so a
/// failed poll asks RunPod about the pod before it waits out POLL_FAILURES. RunPod failing to answer is no evidence
/// either way.
fn check_pod_up(ctx: &Ctx, issue: &str, entry: &PodEntry, job: &str) -> Result<()> {
    let Some(id) = entry.id.as_deref() else {
        return Ok(());
    };
    let what = match ctx.cloud.get_pod(id) {
        Ok(Some(pod)) if matches!(pod.status, PodStatus::Exited | PodStatus::Error | PodStatus::Terminated) => {
            format!("RunPod reports pod {} (id {id}) {:?}", entry.name, pod.status)
        }
        Ok(Some(_)) => return Ok(()),
        Ok(None) => format!("RunPod no longer lists pod {} (id {id})", entry.name),
        Err(e) => {
            ctx.warn(
                issue,
                format!("reading pod {} from RunPod after a failed poll failed: {e}", entry.name),
            );
            return Ok(());
        }
    };
    bail!(
        "{what} while {job} ran: its container is gone, and with it every file on the pod that was not pulled yet; \
         run `harness down {issue}`, since a stopped pod still bills its disk"
    )
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::net::TcpListener;
    use std::thread;

    use super::*;

    #[test]
    fn shell_quote_round_trips_through_bash() {
        let nasty = "it's $HOME `id` \"q\" \\ \n new line";
        let out = Command::new("bash")
            .arg("-c")
            .arg(format!("printf %s {}", shell_quote(nasty)))
            .output()
            .unwrap();
        assert_eq!(String::from_utf8(out.stdout).unwrap(), nasty);
    }

    #[test]
    fn every_script_loads_the_container_env_in_the_kit() {
        let script = remote_script("nvidia-smi");
        assert!(script.starts_with("set -euo pipefail\nsource /root/container_env.sh\n"));
        assert!(script.ends_with("cd /workspace/sm120-kit\nnvidia-smi\n"));
    }

    #[test]
    fn ssh_never_prompts_and_pins_a_per_pod_known_hosts_file() {
        let dir = std::env::temp_dir().join(format!("sm120-ssh-{}", std::process::id()));
        let target = Target {
            host: "203.0.113.7".into(),
            port: 40122,
            known_hosts: dir.join("known_hosts-sm120-x-abc123"),
        };
        let args = |identity: Option<&Path>| {
            let command = ssh_command(&target, identity).unwrap();
            let args: Vec<String> = command.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
            args.join(" ")
        };
        let joined = args(None);
        assert!(joined.contains("BatchMode=yes"), "{joined}");
        assert!(joined.contains("StrictHostKeyChecking=accept-new"), "{joined}");
        assert!(joined.contains("-o ForwardAgent=no"), "{joined}");
        assert!(!joined.contains("IdentitiesOnly"), "{joined}");
        assert!(joined.contains(&format!("UserKnownHostsFile=\"{}\"", target.known_hosts.display())));
        assert!(joined.ends_with("-p 40122 root@203.0.113.7"), "{joined}");
        let keyed = args(Some(Path::new("/Users/me/.ssh/runpod key")));
        assert!(
            keyed.contains("-o IdentitiesOnly=yes -o IdentityFile=\"/Users/me/.ssh/runpod key\""),
            "{keyed}"
        );
        assert!(dir.is_dir());
        let command = ssh_command(&target, None).unwrap();
        assert!(
            command
                .get_envs()
                .any(|(key, value)| key == "RUNPOD_API_KEY" && value.is_none())
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_named_key_that_is_missing_fails_before_any_money_moves() {
        let remote = SshRemote::new(Some(PathBuf::from("/nonexistent/sm120-harness-key")));
        let err = remote.preflight().unwrap_err().to_string();
        assert!(
            err.contains("HARNESS_SSH_KEY names /nonexistent/sm120-harness-key"),
            "{err}"
        );
    }

    fn serve_once(reply: &'static [u8]) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let _ = stream.write_all(reply);
        });
        port
    }

    #[test]
    fn banner_check_wants_an_ssh_greeting() {
        let timeout = Duration::from_secs(2);
        assert!(ssh_banner("127.0.0.1", serve_once(b"SSH-2.0-OpenSSH_9.6\r\n"), timeout));
        assert!(!ssh_banner("127.0.0.1", serve_once(b"HTTP/1.1 400\r\n"), timeout));
        let closed = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        assert!(!ssh_banner("127.0.0.1", closed, timeout));
    }

    #[test]
    fn poll_reply_keeps_only_complete_lines() {
        let running = parse_poll("--harness-poll--\nRUNNING\n{\"a\":1}\n{\"b\":", 10).unwrap();
        assert_eq!(running.state, JobState::Running);
        assert_eq!(running.lines, ["{\"a\":1}"]);
        assert_eq!(running.offset, 18);
        let done = parse_poll("--harness-poll--\nEXIT 3\n", 18).unwrap();
        assert_eq!(
            (done.state, done.lines.len(), done.offset),
            (JobState::Exited(3), 0, 18)
        );
        let dead = parse_poll("--harness-poll--\nDEAD\n", 18).unwrap();
        assert_eq!(dead.state, JobState::Dead);
        assert!(parse_poll("--harness-poll--\nEXIT \n", 0).is_err());
        assert!(parse_poll("RUNNING\n", 0).is_err());
    }

    #[test]
    fn login_shell_output_before_the_sentinel_cannot_break_a_poll() {
        let noisy = "Welcome to Ubuntu 24.04\nEXIT 9\n--harness-poll--\nRUNNING\n{\"a\":1}\n";
        let status = parse_poll(noisy, 0).unwrap();
        assert_eq!((status.state, status.lines.len()), (JobState::Running, 1));
    }

    struct Recorder(RefCell<Vec<String>>);

    impl Remote for Recorder {
        fn preflight(&self) -> Result<()> {
            Ok(())
        }
        fn reachable(&self, _: &Target) -> bool {
            true
        }
        fn run(&self, _: &Target, script: &str) -> Result<String> {
            self.0.borrow_mut().push(script.to_string());
            Ok("STARTED\n".into())
        }
        fn run_streaming(&self, _: &Target, _: &str) -> Result<()> {
            Ok(())
        }
        fn write_file(&self, _: &Target, _: &str, _: &str) -> Result<()> {
            Ok(())
        }
        fn upload(&self, _: &Target, _: &Path, _: &[&str]) -> Result<()> {
            Ok(())
        }
        fn download(&self, _: &Target, _: &[&str], _: &Path, _: Duration) -> Result<()> {
            Ok(())
        }
    }

    fn sh(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[test]
    fn a_hung_child_is_killed_at_the_limit() {
        let started = Instant::now();
        let err = run_bounded(
            sh("sleep 30"),
            b"",
            Output::Discard,
            Duration::from_millis(300),
            "sleeper",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("sleeper ran past its 300ms limit"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn a_child_that_never_reads_its_input_still_hits_the_limit() {
        // More than a pipe buffer of input to a process that never reads it.
        let input = vec![b'x'; 1 << 20];
        let started = Instant::now();
        assert!(
            run_bounded(
                sh("sleep 30"),
                &input,
                Output::Discard,
                Duration::from_millis(300),
                "deaf"
            )
            .is_err()
        );
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn bounded_runs_pass_input_and_capture_output() {
        let input = vec![b'y'; 1 << 20];
        let out = run_bounded(sh("wc -c"), &input, Output::Capture, Duration::from_secs(10), "wc").unwrap();
        assert_eq!(String::from_utf8(out).unwrap().trim(), "1048576");
        let err = run_bounded(sh("exit 3"), b"", Output::Capture, Duration::from_secs(10), "three").unwrap_err();
        assert!(err.to_string().contains("three failed"), "{err}");
    }

    #[test]
    fn a_hung_pipeline_is_killed_whole() {
        let mut first = sh("sleep 30").stdout(Stdio::piped()).spawn().unwrap();
        let mut second = sh("cat >/dev/null")
            .stdin(first.stdout.take().unwrap())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let err = wait_within(&mut [&mut first, &mut second], Duration::from_millis(300), "pipe").unwrap_err();
        assert!(err.to_string().contains("pipe ran past"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));
        assert!(first.try_wait().unwrap().is_some() && second.try_wait().unwrap().is_some());
    }

    #[test]
    fn a_download_skips_missing_paths_and_accepts_a_changed_file() {
        let command = download_command(&["results/x/a1", "results/x/it's"]);
        assert!(
            command.starts_with("cd /workspace/sm120-kit || exit 2; set --; "),
            "{command}"
        );
        assert!(
            command.contains("for p in 'results/x/a1' 'results/x/it'\\''s'; do"),
            "{command}"
        );
        assert!(
            command.contains("tar --warning=no-file-changed -czf - -T /dev/null \"$@\""),
            "{command}"
        );
        assert!(command.ends_with("[ \"$rc\" -le 1 ]"), "{command}");
    }

    #[test]
    fn jobs_start_detached_and_record_their_exit_code() {
        let recorder = Recorder(RefCell::new(Vec::new()));
        let target = Target {
            host: "h".into(),
            port: 1,
            known_hosts: PathBuf::from("k"),
        };
        start_job(
            &recorder,
            &target,
            "vllm-47990",
            "setup",
            "bash issues/vllm-47990/setup.sh",
        )
        .unwrap();
        let script = recorder.0.borrow()[0].clone();
        assert!(
            script.contains("rm -f results/vllm-47990/jobs/setup.exit results/vllm-47990/jobs/setup.pid\n"),
            "{script}"
        );
        assert!(
            script.contains("setsid nohup bash -c 'echo $$ > results/vllm-47990/jobs/setup.pid.tmp"),
            "{script}"
        );
        assert!(
            script.contains("(\nbash issues/vllm-47990/setup.sh\n)\necho $?"),
            "{script}"
        );
        assert!(
            script.contains("> results/vllm-47990/jobs/setup.log 2>&1 < /dev/null &\n"),
            "{script}"
        );
    }

    /// Runs the job scripts in local bash in a temp dir standing in for the kit directory, with a `setsid` shim
    /// where the system has none (macOS).
    struct LocalBash {
        dir: PathBuf,
        path: String,
    }

    impl LocalBash {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("sm120-jobs-{tag}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let shims = dir.join(".shims");
            fs::create_dir_all(&shims).unwrap();
            let mut path = std::env::var("PATH").unwrap_or_default();
            if !Command::new("sh")
                .args(["-c", "command -v setsid"])
                .status()
                .unwrap()
                .success()
            {
                let shim = shims.join("setsid");
                fs::write(&shim, "#!/bin/sh\nexec \"$@\"\n").unwrap();
                Command::new("chmod").arg("+x").arg(&shim).status().unwrap();
                path = format!("{}:{path}", shims.display());
            }
            Self { dir, path }
        }

        fn poll_until_done(&self, job: &str, tail: &str) -> (JobState, Vec<String>, u64) {
            let target = local_target();
            let (mut lines, mut offset) = (Vec::new(), 0);
            for _ in 0..200 {
                let status = poll_job(self, &target, "x-1", job, tail, offset).unwrap();
                lines.extend(status.lines);
                offset = status.offset;
                if status.state != JobState::Running {
                    return (status.state, lines, offset);
                }
                thread::sleep(Duration::from_millis(50));
            }
            panic!("job {job} never finished");
        }

        fn pid(&self, job: &str) -> String {
            fs::read_to_string(self.dir.join(format!("results/x-1/jobs/{job}.pid")))
                .unwrap()
                .trim()
                .to_string()
        }
    }

    impl Drop for LocalBash {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn local_target() -> Target {
        Target {
            host: "local".into(),
            port: 0,
            known_hosts: PathBuf::from("unused"),
        }
    }

    impl Remote for LocalBash {
        fn preflight(&self) -> Result<()> {
            Ok(())
        }
        fn reachable(&self, _: &Target) -> bool {
            true
        }
        fn run(&self, _: &Target, script: &str) -> Result<String> {
            let out = Command::new("bash")
                .arg("-c")
                .arg(format!("set -euo pipefail\n{script}"))
                .current_dir(&self.dir)
                .env("PATH", &self.path)
                .output()?;
            ensure!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
            Ok(String::from_utf8(out.stdout)?)
        }
        fn run_streaming(&self, _: &Target, _: &str) -> Result<()> {
            Ok(())
        }
        fn write_file(&self, _: &Target, _: &str, _: &str) -> Result<()> {
            Ok(())
        }
        fn upload(&self, _: &Target, _: &Path, _: &[&str]) -> Result<()> {
            Ok(())
        }
        fn download(&self, _: &Target, _: &[&str], _: &Path, _: Duration) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_real_job_streams_a_file_that_grows_in_partial_lines() {
        let bash = LocalBash::new("grow");
        let writer = "f=results/x-1/out.jsonl; mkdir -p results/x-1\n\
                      printf '{\"phase\":\"a\"}\\n{\"pha' >> $f; sleep 0.4\n\
                      printf 'se\":\"b\"}\\n' >> $f; sleep 0.4\n\
                      printf '{\"verdict\":\"OK\"}\\n' >> $f\nexit 3";
        start_job(&bash, &local_target(), "x-1", "grow", writer).unwrap();
        let (state, lines, offset) = bash.poll_until_done("grow", "results/x-1/out.jsonl");
        assert_eq!(state, JobState::Exited(3));
        assert_eq!(
            lines,
            ["{\"phase\":\"a\"}", "{\"phase\":\"b\"}", "{\"verdict\":\"OK\"}"]
        );
        assert_eq!(
            offset,
            fs::metadata(bash.dir.join("results/x-1/out.jsonl")).unwrap().len()
        );
    }

    #[test]
    fn a_real_job_killed_without_an_exit_code_reads_as_dead() {
        let bash = LocalBash::new("dead");
        start_job(&bash, &local_target(), "x-1", "victim", "sleep 30").unwrap();
        let pid = bash.pid("victim");
        let first = poll_job(&bash, &local_target(), "x-1", "victim", "/dev/null", 0).unwrap();
        assert_eq!(first.state, JobState::Running);
        assert!(Command::new("kill").args(["-9", &pid]).status().unwrap().success());
        let (state, _, _) = bash.poll_until_done("victim", "/dev/null");
        assert_eq!(state, JobState::Dead);
    }

    #[test]
    fn a_stale_pid_that_now_names_another_process_reads_as_dead() {
        let bash = LocalBash::new("reused");
        let jobs = bash.dir.join("results/x-1/jobs");
        fs::create_dir_all(&jobs).unwrap();
        // As after a container restart: the old pid now belongs to a live process that is not the job.
        fs::write(jobs.join("old.pid"), format!("{}\n", std::process::id())).unwrap();
        let status = poll_job(&bash, &local_target(), "x-1", "old", "/dev/null", 0).unwrap();
        assert_eq!(status.state, JobState::Dead);
        start_job(&bash, &local_target(), "x-1", "next", "true").unwrap();
        let (state, _, _) = bash.poll_until_done("next", "/dev/null");
        assert_eq!(state, JobState::Exited(0));
    }

    #[test]
    fn a_real_job_refuses_to_start_beside_a_running_one() {
        let bash = LocalBash::new("busy");
        start_job(&bash, &local_target(), "x-1", "first", "sleep 30").unwrap();
        let err = start_job(&bash, &local_target(), "x-1", "second", "true").unwrap_err();
        let pid = bash.pid("first");
        assert!(
            err.to_string()
                .contains(&format!("job first (pid {pid}) of issue x-1 still runs")),
            "{err}"
        );
        Command::new("kill").arg(bash.pid("first")).status().unwrap();
        bash.poll_until_done("first", "/dev/null");
        start_job(&bash, &local_target(), "x-1", "second", "true").unwrap();
        let (state, _, _) = bash.poll_until_done("second", "/dev/null");
        assert_eq!(state, JobState::Exited(0));
    }
}
