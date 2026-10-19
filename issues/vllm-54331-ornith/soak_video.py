"""Saturates a vLLM OpenAI server with multi-turn video windows for a fixed time.

The report's workload is offline batch video inference: several windows per video, each a turn over the same system
prompt and video, up to 70 at once. Here request n is window n % WINDOWS of video (n // WINDOWS) % len(videos), so
about nine videos are in flight at a time. Each window extends the previous one's prompt, so the later windows of a
video reuse its cached blocks (blocks are about 2K tokens on this model, so only prompts past one block hit), and a
new video starts every few seconds, which keeps the vision encoder busy for the whole soak instead of only the first
minute.

Exits 1 if any request failed, so the harness records the variant as POST_FAIL, and 2 if the probe shows the clip
never reached the model.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

WINDOWS = 8
MAX_TOKENS = 512

SYSTEM_PROMPT = " ".join(
    [
        "You are an event detector for fixed security cameras.",
        "For each clip, list every event you see as one JSON object per line with the keys start_s, end_s, label,",
        "confidence and description. Labels are person_enters, person_leaves, vehicle_moves, object_left,",
        "object_removed, door_opens, door_closes, light_change and other. Report times in seconds from the start of",
        "the clip, to one decimal place. Describe only what is visible; never guess at identities.",
    ]
    * 8
)

PRIOR_ANSWER = "\n".join(
    json.dumps({"start_s": i * 0.5, "end_s": i * 0.5 + 0.4, "label": "other", "confidence": 0.5,
                "description": "a rectangle moves across the frame from left to right"})
    for i in range(4)
)


def messages(video: Path, window: int, with_video: bool = True) -> list[dict]:
    first = [{"type": "text", "text": "List the events in this clip."}]
    if with_video:
        first.insert(0, {"type": "video_url", "video_url": {"url": video.as_uri()}})
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": first}]
    for w in range(1, window + 1):
        msgs.append({"role": "assistant", "content": PRIOR_ANSWER})
        msgs.append({"role": "user", "content": f"Window {w}: list only the events after second {w * 0.5:.1f}."})
    return msgs


async def complete(session: aiohttp.ClientSession, url: str, model: str, msgs: list[dict]) -> dict:
    body = {"model": model, "messages": msgs, "max_tokens": MAX_TOKENS, "temperature": 0.7, "ignore_eos": True}
    async with session.post(url, json=body) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
        return json.loads(text)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--minutes", type=float, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        print(f"no clips in {args.videos}", file=sys.stderr)
        return 2
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # A server that dropped the clip would answer every request and turn this into a second text soak.
        with_clip = await complete(session, url, args.model, messages(videos[0], 0))
        without = await complete(session, url, args.model, messages(videos[0], 0, with_video=False))
        clip_tokens = with_clip["usage"]["prompt_tokens"] - without["usage"]["prompt_tokens"]
        print(f"one clip adds {clip_tokens} prompt tokens", flush=True)
        if clip_tokens < 256:
            print("the clip did not reach the model", file=sys.stderr)
            return 2

        state = {"next": 0, "ok": 0, "fail": 0, "out_tokens": 0, "prompt_tokens": 0, "cached_tokens": 0}
        errors: dict[str, int] = {}
        start = time.monotonic()
        end = start + args.minutes * 60

        async def worker() -> None:
            while time.monotonic() < end:
                n = state["next"]
                state["next"] += 1
                video = videos[(n // WINDOWS) % len(videos)]
                try:
                    reply = await complete(session, url, args.model, messages(video, n % WINDOWS))
                except Exception as exc:  # every failure counts; the report's surfaces end the engine
                    state["fail"] += 1
                    key = f"{type(exc).__name__}: {exc}"[:200]
                    errors[key] = errors.get(key, 0) + 1
                    continue
                usage = reply["usage"]
                state["ok"] += 1
                state["out_tokens"] += usage["completion_tokens"]
                state["prompt_tokens"] += usage["prompt_tokens"]
                state["cached_tokens"] += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0

        async def report() -> None:
            last_t, last_out = start, 0
            with args.out.open("a") as log:
                while True:
                    await asyncio.sleep(60)
                    now = time.monotonic()
                    line = {
                        "elapsed_s": round(now - start),
                        "ok": state["ok"],
                        "fail": state["fail"],
                        "out_tok_s": round((state["out_tokens"] - last_out) / (now - last_t)),
                        "mean_prompt_tokens": round(state["prompt_tokens"] / max(state["ok"], 1)),
                        "cached_share": round(state["cached_tokens"] / max(state["prompt_tokens"], 1), 3),
                    }
                    last_t, last_out = now, state["out_tokens"]
                    print(json.dumps(line), flush=True)
                    log.write(json.dumps(line) + "\n")
                    log.flush()

        reporter = asyncio.create_task(report())
        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
        reporter.cancel()

    elapsed = time.monotonic() - start
    print(f"soak finished: {state['ok']} ok, {state['fail']} failed in {elapsed / 60:.1f} min, "
          f"{state['out_tokens'] / elapsed:.0f} output tok/s", flush=True)
    for key, count in sorted(errors.items(), key=lambda kv: -kv[1])[:5]:
        print(f"{count}x {key}", flush=True)
    return 1 if state["fail"] else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
