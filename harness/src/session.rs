use std::fs;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};

pub const KIT_DIR: &str = "/workspace/sm120-kit";
const UPLOAD_EXCLUDES: [&str; 3] = [".DS_Store", "harness-ledger.json*", ".harness-ledger.lock"];

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
    fn download(&self, target: &Target, path: &str, local_root: &Path) -> Result<()>;
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

pub struct SshRemote;

fn ssh_command(target: &Target) -> Result<Command> {
    if let Some(dir) = target.known_hosts.parent() {
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let mut command = Command::new("ssh");
    command
        .args(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"])
        .arg("-o")
        .arg(format!("UserKnownHostsFile=\"{}\"", target.known_hosts.display()))
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

fn run_script(target: &Target, script: &str, stdout: Stdio) -> Result<Vec<u8>> {
    let mut child = ssh_command(target)?
        .arg("bash -s")
        .stdin(Stdio::piped())
        .stdout(stdout)
        .spawn()
        .context("starting ssh")?;
    child
        .stdin
        .take()
        .context("ssh stdin")?
        .write_all(remote_script(script).as_bytes())?;
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "remote script failed ({}) on {}:{}",
        output.status,
        target.host,
        target.port
    );
    Ok(output.stdout)
}

impl Remote for SshRemote {
    fn preflight(&self) -> Result<()> {
        let status = Command::new("ssh-add")
            .arg("-l")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .context("running ssh-add")?;
        ensure!(
            status.success(),
            "ssh-agent holds no key (ssh-add -l failed); run ssh-add first: ssh runs with BatchMode and cannot ask for a passphrase"
        );
        Ok(())
    }

    fn reachable(&self, target: &Target) -> bool {
        ssh_banner(&target.host, target.port, Duration::from_secs(5))
    }

    fn run(&self, target: &Target, script: &str) -> Result<String> {
        let stdout = run_script(target, script, Stdio::piped())?;
        Ok(String::from_utf8_lossy(&stdout).into_owned())
    }

    fn run_streaming(&self, target: &Target, script: &str) -> Result<()> {
        run_script(target, script, Stdio::inherit()).map(drop)
    }

    fn write_file(&self, target: &Target, path: &str, content: &str) -> Result<()> {
        let full = format!("{KIT_DIR}/{path}");
        let dir = full.rsplit_once('/').map_or(KIT_DIR, |(dir, _)| dir);
        let mut child = ssh_command(target)?
            .arg(format!("mkdir -p {} && cat > {}", shell_quote(dir), shell_quote(&full)))
            .stdin(Stdio::piped())
            .spawn()
            .context("starting ssh")?;
        child.stdin.take().context("ssh stdin")?.write_all(content.as_bytes())?;
        let status = child.wait()?;
        ensure!(status.success(), "writing {full} on the pod failed ({status})");
        Ok(())
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
        let ssh = ssh_command(target)?
            .arg(format!("mkdir -p {KIT_DIR} && tar --no-same-owner -xzf - -C {KIT_DIR}"))
            .stdin(Stdio::from(archive))
            .status()
            .context("starting ssh")?;
        let tar = tar.wait()?;
        ensure!(
            tar.success() && ssh.success(),
            "copying {paths:?} to the pod failed (tar {tar}, ssh {ssh})"
        );
        Ok(())
    }

    fn download(&self, target: &Target, path: &str, local_root: &Path) -> Result<()> {
        let mut ssh = ssh_command(target)?
            .arg(format!("tar -czf - -C {KIT_DIR} {}", shell_quote(path)))
            .stdout(Stdio::piped())
            .spawn()
            .context("starting ssh")?;
        let archive = ssh.stdout.take().context("ssh stdout")?;
        fs::create_dir_all(local_root)?;
        let tar = Command::new("tar")
            .args(["-xzf", "-", "-C"])
            .arg(local_root)
            .stdin(Stdio::from(archive))
            .status()
            .context("starting tar")?;
        let ssh = ssh.wait()?;
        ensure!(
            tar.success() && ssh.success(),
            "pulling {path} failed (tar {tar}, ssh {ssh})"
        );
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct JobStatus {
    pub exit: Option<i32>,
    pub lines: Vec<String>,
    pub offset: u64,
}

pub fn job_dir(issue: &str) -> String {
    format!("results/{issue}/jobs")
}

/// Starts `command` detached from the SSH session, so a dropped connection never kills a run. The exit code is
/// written by rename once the command ends, so a poll never reads a half-written file.
pub fn start_job(remote: &dyn Remote, target: &Target, issue: &str, job: &str, command: &str) -> Result<()> {
    let dir = job_dir(issue);
    let body = format!("{command}\necho $? > {dir}/{job}.exit.tmp && mv {dir}/{job}.exit.tmp {dir}/{job}.exit");
    let script = format!(
        "mkdir -p {dir}\nrm -f {dir}/{job}.exit\nsetsid nohup bash -c {} > {dir}/{job}.log 2>&1 < /dev/null &",
        shell_quote(&body)
    );
    remote.run(target, &script).map(drop)
}

/// Reports whether the job has exited and returns the complete lines added to `tail_path` since `offset`. The
/// exit check comes first, so once it reports an exit the tail holds everything the job wrote.
pub fn poll_job(
    remote: &dyn Remote,
    target: &Target,
    issue: &str,
    job: &str,
    tail_path: &str,
    offset: u64,
) -> Result<JobStatus> {
    let dir = job_dir(issue);
    let script = format!(
        "if [[ -f {dir}/{job}.exit ]]; then echo \"EXIT $(cat {dir}/{job}.exit)\"; else echo RUNNING; fi\n\
         tail -c +{} {tail_path} 2>/dev/null || true",
        offset + 1
    );
    parse_poll(&remote.run(target, &script)?, offset)
}

pub fn parse_poll(output: &str, offset: u64) -> Result<JobStatus> {
    let (head, rest) = output.split_once('\n').unwrap_or((output, ""));
    let exit = match head.strip_prefix("EXIT ") {
        Some(code) => Some(code.trim().parse().with_context(|| format!("bad exit line {head:?}"))?),
        None if head == "RUNNING" => None,
        None => bail!("unexpected poll reply {head:?}"),
    };
    let complete = rest.rfind('\n').map_or("", |end| &rest[..=end]);
    Ok(JobStatus {
        exit,
        lines: complete
            .lines()
            .filter(|line| !line.is_empty())
            .map(str::to_string)
            .collect(),
        offset: offset + complete.len() as u64,
    })
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
        let command = ssh_command(&target).unwrap();
        let args: Vec<String> = command.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let joined = args.join(" ");
        assert!(joined.contains("BatchMode=yes"), "{joined}");
        assert!(joined.contains("StrictHostKeyChecking=accept-new"), "{joined}");
        assert!(joined.contains(&format!("UserKnownHostsFile=\"{}\"", target.known_hosts.display())));
        assert!(joined.ends_with("-p 40122 root@203.0.113.7"), "{joined}");
        assert!(dir.is_dir());
        fs::remove_dir_all(dir).unwrap();
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
        let running = parse_poll("RUNNING\n{\"a\":1}\n{\"b\":", 10).unwrap();
        assert_eq!(running.exit, None);
        assert_eq!(running.lines, ["{\"a\":1}"]);
        assert_eq!(running.offset, 18);
        let done = parse_poll("EXIT 3\n", 18).unwrap();
        assert_eq!((done.exit, done.lines.len(), done.offset), (Some(3), 0, 18));
        assert!(parse_poll("EXIT \n", 0).is_err());
        assert!(parse_poll("Welcome to Ubuntu\n", 0).is_err());
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
            Ok(String::new())
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
        fn download(&self, _: &Target, _: &str, _: &Path) -> Result<()> {
            Ok(())
        }
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
            script.contains("rm -f results/vllm-47990/jobs/setup.exit\n"),
            "{script}"
        );
        assert!(
            script.contains("setsid nohup bash -c 'bash issues/vllm-47990/setup.sh\necho $?"),
            "{script}"
        );
        assert!(
            script.ends_with("> results/vllm-47990/jobs/setup.log 2>&1 < /dev/null &"),
            "{script}"
        );
    }
}
