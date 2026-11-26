"""Runs QualityBench's CLI with milabench's adjustments, in QualityBench's own environment (standalone: it
imports nothing from milabench). The third-party install stays untouched; each adjustment is opt-in:

  QB_EVALPLUS_IMAGE        the evalplus image to use instead of QualityBench's (its own is x86_64 only)
  QB_MAX_TOKENS_FLOOR      raise every chat request's max_tokens to at least this, thinking off too.
                           QualityBench hard-codes LiveCodeBench's answer limit at 4096 tokens with thinking off,
                           and models that reason in their visible answer (Qwen3, Nemotron) are cut there.
  QB_PREFLIGHT_RETRY       with thinking on, when the "Reply with the single word OK" preflight comes back without
                           reasoning, ask once more with a question worth thinking about before failing: models
                           that decide per prompt whether to think (Gemma 4 E4B) skip reasoning on the trivial one.

Always: a time-to-solution window, QB_REPLY_WINDOW s per reply (default 3600, quality.py's reply_window). A reply not
back by then is an empty answer with finish_reason "timeout": scored as wrong, not retried, and counted in state.json
(caps.<task>.timeouts). QualityBench alone would retry it from scratch (lm-eval up to 3 times, LiveCodeBench 3 attempts). A slow dense model writing a 32k-token reply (~8 tokens/s per request with 16 in
flight) needs more than an hour, and after three timeouts the task fails.

With QB_RESTREAM=1: plain chat requests are streamed from the server and rebuilt into the same completion for
the harness, so a timed-out reply leaves its partial text for the timing log (the harness still gets nothing).

Always: one timing line per reply, in the run's sidecar timing/<task>.jsonl (seconds to headers, first bytes and
end; requests in flight; finish reason; tokens), for the distribution of time to solution per model and problem.

Always: each task's generated tokens (the replies' usage.completion_tokens) are counted next to QualityBench's
own reply counts, in its state.json (caps.<task>.tokens), for the report's tokens-to-solution.

  python qb_launch.py <qualitybench argv...>
"""

import json
import os
import sys


def _max_tokens_floor(floor: int):
    import json

    from qualitybench import proxy

    rewrite = proxy.Proxy.rewrite

    def floored(self, path, body):
        body = rewrite(self, path, body)
        if not path.rstrip("/").endswith("/chat/completions"):
            return body
        try:
            d = json.loads(body)
        except ValueError:
            return body
        d.pop("max_completion_tokens", None)
        d["max_tokens"] = max(d.get("max_tokens") or 0, floor)
        return json.dumps(d).encode()

    proxy.Proxy.rewrite = floored


PREFLIGHT_RETRY_PROMPT = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Answer with the amount only."
)


def _preflight_retry():
    from qualitybench import runner

    preflight = runner.preflight

    def retried(ctx, client):
        try:
            return preflight(ctx, client)
        except runner.TaskFailed as e:
            if ctx.thinking != "on" or "thinking is not on" not in str(e):
                raise
        runner.log_to(ctx, "preflight: no reasoning on the trivial prompt, asking a question worth thinking about")
        r = client.chat(ctx.model, [{"role": "user", "content": PREFLIGHT_RETRY_PROMPT}],
                        max_tokens=4096, temperature=0, log=lambda m: runner.log_to(ctx, m))
        m = r["choices"][0]["message"]
        reasoning = m.get("reasoning") or m.get("reasoning_content")
        runner.log_to(ctx, f"preflight retry content: {m.get('content')!r} reasoning: {'yes' if reasoning else 'no'}")
        if not reasoning:
            raise runner.TaskFailed("preflight: thinking is not on (no reasoning text, on a trivial prompt nor on a "
                                    "question worth thinking about)")

    runner.preflight = retried


def _count_tokens():
    import json

    from qualitybench import proxy

    tally = proxy.Proxy.tally

    def counted(self, body):
        tally(self, body)
        if b'"finish_reason": "timeout"' in body or b'"finish_reason":"timeout"' in body:
            with self._lock:
                self._stats["timeouts"] = self._stats.get("timeouts", 0) + 1
        tokens = 0
        text = body.decode("utf-8", "replace")
        docs = [text] if text.lstrip().startswith("{") else [l[5:] for l in text.splitlines() if l.startswith("data:")]
        for doc in docs:
            try:
                usage = json.loads(doc).get("usage") or {}
            except (ValueError, AttributeError):
                continue
            tokens = max(tokens, int(usage.get("completion_tokens") or 0))
        if tokens:
            with self._lock:
                self._stats["tokens"] = self._stats.get("tokens", 0) + tokens

    proxy.Proxy.tally = counted


def _lcb_timeout(seconds: int):
    """QualityBench reads its helper scripts from its package's helpers folder at call time: hand it a copy whose
    LiveCodeBench generator waits longer per reply."""
    import shutil
    import tempfile
    from pathlib import Path

    from qualitybench import envs

    pkg_file = envs.pkg_file
    src = pkg_file("helpers")
    gen = (src / "lcb_gen.py").read_text()
    if "timeout=3600" not in gen:
        return
    dst = Path(tempfile.mkdtemp(prefix="qb-helpers-")) / "helpers"
    shutil.copytree(src, dst)
    (dst / "lcb_gen.py").write_text(gen.replace("timeout=3600", f"timeout={int(seconds)}"))

    def patched(*parts):
        return dst.joinpath(*parts[1:]) if parts and parts[0] == "helpers" else pkg_file(*parts)

    envs.pkg_file = patched


TIMEOUT_REPLY = "timeout"  # finish_reason of a reply that ran out of the time-to-solution window

# with QB_RESTREAM=1, a streamed reply whose last LOOP_MIN characters compress below LOOP_CUT (zlib ratio; normal
# answers 0.25-0.4, loops < 0.05) on two checks LOOP_EVERY characters apart is cut there: the harness gets it as
# stopped at the token cap (finish_reason "length", the text so far), as it would have been, minus the wait.
# QB_LOOP_CUT=0 turns it off.
LOOP_CUT = float(os.environ.get("QB_LOOP_CUT", "0.08"))
LOOP_MIN, LOOP_EVERY = 20000, 4000

import threading as _threading

_LOCAL = _threading.local()  # the proxy handler thread: restream flag, partial text and first-token time of its reply


def _restream():
    """Ask the server to stream every plain chat request (stream, stream_options.include_usage) and hand the harness
    the same non-streamed completion, rebuilt from the chunks (_reply_window does the rebuilding). So a reply past
    the window leaves its partial text (for the timing log only: the harness still gets an empty answer), and the
    timing log gets the real first-token time. Requests with tools, n > 1 or logprobs pass through untouched.
    Opt-in (QB_RESTREAM=1) until checked against a real server."""
    from qualitybench import proxy

    rewrite = proxy.Proxy.rewrite

    def streamed(self, path, body):
        body = rewrite(self, path, body)
        _LOCAL.restream = False
        if not path.rstrip("/").endswith("/chat/completions"):
            return body
        try:
            d = json.loads(body)
        except ValueError:
            return body
        if d.get("stream") or d.get("tools") or (d.get("n") or 1) > 1 or d.get("logprobs"):
            return body
        d["stream"] = True
        d["stream_options"] = {"include_usage": True}
        _LOCAL.restream = True
        return json.dumps(d).encode()

    proxy.Proxy.rewrite = streamed


def _reply_window(window: int):
    """The time-to-solution window: a reply not back within `window` s scores as wrong, at once, with no retry.

    QualityBench's proxy reads each reply from the server under qualitybench.http.TIMEOUT. Past the window, the
    connection to the server is shut (vLLM aborts the generation) and the proxy forwards, in its place, an empty
    chat completion with finish_reason "timeout": every harness takes it as an empty answer (wrong) and does not
    retry. lm-eval's and LiveCodeBench's own client timeouts are pushed past the window so the proxy decides first.
    Timeouts are counted in state.json (caps.<task>.timeouts) next to the replies cut at the token cap."""
    import io
    import time
    import zlib

    from qualitybench import context
    from qualitybench import http as qhttp

    qhttp.TIMEOUT = window
    connect = qhttp.connect

    class TimedOut:  # what http.client's response looks like to the proxy
        status = 200

        def __init__(self, model, stream):
            doc = {"id": "qb-timeout", "object": "chat.completion", "created": int(time.time()), "model": model,
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                "finish_reason": TIMEOUT_REPLY}],
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            if stream:
                doc["object"] = "chat.completion.chunk"
                doc["choices"][0]["delta"] = doc["choices"][0].pop("message")
                body = f"data: {json.dumps(doc)}\n\ndata: [DONE]\n\n".encode()
                self._type = "text/event-stream"
            else:
                body = json.dumps(doc).encode()
                self._type = "application/json"
            self._buf = io.BytesIO(body)

        def getheaders(self):
            return [("Content-Type", self._type)]

        def read1(self, n=-1):
            return self._buf.read1(n)

        def read(self, n=-1):
            return self._buf.read(n)

    class Rebuilt(TimedOut):  # a plain chat completion from the chunks of a streamed one
        def __init__(self, doc, status=200):
            self.status = status
            self._type = "application/json"
            self._buf = io.BytesIO(json.dumps(doc).encode())

    def rebuild(r, conn, sent):
        content, reasoning, rkey = [], [], None
        fin, usage, head = None, None, {}
        deadline = sent["t0"] + window
        _LOCAL.partial, _LOCAL.t_first_token, _LOCAL.loop_cut = None, None, None
        chars, check, low = 0, LOOP_MIN, 0
        try:
            while True:
                left = deadline - time.time()
                if left <= 0:
                    raise TimeoutError
                if conn.sock is not None:
                    conn.sock.settimeout(left)
                line = r.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    d = json.loads(data)
                except ValueError:
                    continue
                if "error" in d and not d.get("choices"):
                    return Rebuilt(d, 500)
                if not head:
                    head = {k: d.get(k) for k in ("id", "created", "model", "system_fingerprint") if d.get(k) is not None}
                usage = d.get("usage") or usage
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    if delta.get("content"):
                        content.append(delta["content"])
                    for k in ("reasoning_content", "reasoning"):
                        if delta.get(k):
                            rkey = rkey or k
                            reasoning.append(delta[k])
                    if (delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")) \
                            and _LOCAL.t_first_token is None:
                        _LOCAL.t_first_token = time.time()
                    fin = ch.get("finish_reason") or fin
                    chars += len(delta.get("content") or "") + len(delta.get("reasoning_content") or delta.get("reasoning") or "")
                if LOOP_CUT and chars >= check and not fin:
                    check = chars + LOOP_EVERY
                    tail = ("".join(reasoning) + "".join(content))[-LOOP_MIN:].encode()
                    ratio = len(zlib.compress(tail)) / len(tail)
                    low = low + 1 if ratio < LOOP_CUT else 0
                    if low >= 2:  # two checks in a row: repeating itself, it will run to the cap
                        qhttp.hard_close(conn)  # the server drops the generation
                        _LOCAL.loop_cut = round(ratio, 4)
                        fin = "length"  # what the cap would have given it, a few minutes or hours later
                        break
        except (TimeoutError, OSError):
            qhttp.hard_close(conn)  # the server sees the disconnect and drops the generation
            _LOCAL.partial = ("".join(content), "".join(reasoning))  # the timing log only: the answer is empty
            return TimedOut(sent["model"], False)
        msg = {"role": "assistant", "content": "".join(content)}
        if rkey:
            msg[rkey] = "".join(reasoning)
        doc = {**head, "object": "chat.completion",
               "choices": [{"index": 0, "message": msg, "finish_reason": fin or "stop"}]}
        doc.setdefault("model", sent["model"])
        if usage:
            doc["usage"] = usage
        return Rebuilt(doc)

    def windowed(url, timeout):
        conn, path = connect(url, timeout)
        request, getresponse = conn.request, conn.getresponse
        sent = {}

        def req(method, url, body=None, headers=None, **kw):
            if method == "POST" and url.rstrip("/").endswith("/chat/completions") and body:
                try:
                    d = json.loads(body)
                    restream = getattr(_LOCAL, "restream", False)
                    _LOCAL.restream = False
                    # stream: what the harness asked for (a restreamed request was plain)
                    sent.update(model=d.get("model"), stream=bool(d.get("stream")) and not restream,
                                restream=restream, t0=time.time())
                except ValueError:
                    pass
            return request(method, url, body, headers or {}, **kw)

        def resp():
            try:
                r = getresponse()
            except TimeoutError:
                if not sent:
                    raise
                qhttp.hard_close(conn)  # the server sees the disconnect and drops the generation
                return TimedOut(sent["model"], sent["stream"])
            if sent.get("restream") and r.status == 200:
                return rebuild(r, conn, sent)
            return r

        conn.request, conn.getresponse = req, resp
        return conn, path

    qhttp.connect = windowed

    sh = context.Ctx.sh

    def patched(self, cmd, *args, **kwargs):  # lm-eval: --model_args ...,timeout=3600
        cmd = [a.replace("timeout=3600", f"timeout={window + 600}") if isinstance(a, str) else a for a in cmd]
        return sh(self, cmd, *args, **kwargs)

    context.Ctx.sh = patched


def _reply_timings():
    """One JSON line per chat reply the proxy forwards upstream, in the run's sidecar: timing/<task>.jsonl.

    t_start (request sent), t_headers (response headers), t_first (first body bytes), t_end (body done or failed),
    in_flight (upstream chat requests open when this one started, itself included), finish_reason ("timeout": past
    the window; "error": the upstream failed), completion/reasoning tokens from usage when the server gives them,
    prompt_sha1 (of the request's messages: the same problem has the same hash across models and runs), chars and
    reasoning_chars, loop_ratio (zlib ratio of the last 20k characters of reasoning + answer: < 0.1 is a loop), and
    for a reply cut at the token cap, the last 4k characters of each (tail_content, tail_reasoning).
    For the report's seconds-per-reply distribution and the accuracy-vs-window curve. Requests outside a task
    (preflight, speed probe) are not written."""
    import hashlib
    import threading
    import time
    import zlib

    LOOP_TAIL, KEEP_TAIL = 20000, 4096

    from qualitybench import http as qhttp
    from qualitybench import proxy, runner

    lock = threading.Lock()
    local = threading.local()  # the proxy handler thread: its request's timing, filled in as it goes
    current = {"side": None, "task": None}
    open_n = [0]

    run_task = runner.run_task

    def tasked(ctx, task, client):
        current.update(side=ctx.side, task=task)
        try:
            return run_task(ctx, task, client)
        finally:
            current.update(side=None, task=None)

    runner.run_task = tasked

    track, untrack = qhttp.track, qhttp.untrack

    handler = proxy.Proxy._handler

    def marked(self):  # the proxy's forwarding (not the in-process client, which tracks its connections too)
        H = handler(self)

        class Fwd(H):
            def _fwd(self, body):
                local.fwd = True
                try:
                    return super()._fwd(body)
                finally:
                    local.fwd = False

        return Fwd

    proxy.Proxy._handler = marked

    def tracked(conn):
        track(conn)
        if current["task"] is None or not getattr(local, "fwd", False):
            return
        local.rec = rec = {"task": current["task"], "_side": current["side"]}
        request, getresponse = conn.request, conn.getresponse

        def req(method, url, body=None, headers=None, **kw):
            if method == "POST" and url.rstrip("/").endswith("/chat/completions") and body:
                try:
                    msgs = json.loads(body).get("messages")
                    rec["prompt_sha1"] = hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()[:16]
                except (ValueError, AttributeError):
                    pass
                with lock:
                    open_n[0] += 1
                    rec["in_flight"] = open_n[0]
                rec["t_start"] = time.time()
            return request(method, url, body, headers, **kw)

        def resp():
            r = getresponse()
            rec["t_headers"] = time.time()
            for name in ("read1", "read"):
                f = getattr(r, name, None)
                if f is None:
                    continue

                def first(n=-1, _f=f):
                    chunk = _f(n)
                    if chunk and "t_first" not in rec:
                        rec["t_first"] = time.time()
                    return chunk

                try:
                    setattr(r, name, first)
                except AttributeError:
                    pass
            return r

        conn.request, conn.getresponse = req, resp

    def untracked(conn):
        untrack(conn)
        rec = getattr(local, "rec", None)
        local.rec = None
        if not rec or "t_start" not in rec:
            return
        with lock:
            open_n[0] -= 1
        rec["t_end"] = time.time()
        rec.setdefault("finish_reason", "error")
        side = rec.pop("_side")
        try:
            path = side / "timing" / f"{rec['task']}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with lock, open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    qhttp.track, qhttp.untrack = tracked, untracked

    tally = proxy.Proxy.tally

    def noted(self, body):  # the reply's finish_reason and usage, for its timing line (same handler thread)
        tally(self, body)
        rec = getattr(local, "rec", None)
        if rec is None:
            return
        text = body.decode("utf-8", "replace")
        docs = [text] if text.lstrip().startswith("{") else [l[5:] for l in text.splitlines() if l.startswith("data:")]
        content, reasoning = [], []
        for doc in docs:
            try:
                d = json.loads(doc)
            except ValueError:
                continue
            for ch in d.get("choices") or []:
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
                m = ch.get("message") or ch.get("delta") or {}
                content.append(m.get("content") or "")
                reasoning.append(m.get("reasoning") or m.get("reasoning_content") or "")
            u = d.get("usage") or {}
            if u.get("completion_tokens"):
                rec["completion_tokens"] = u["completion_tokens"]
            r = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if r:
                rec["reasoning_tokens"] = r
        content, reasoning = "".join(content), "".join(reasoning)
        partial = getattr(_LOCAL, "partial", None)
        if rec.get("finish_reason") == TIMEOUT_REPLY and partial:  # what had streamed in before the window closed
            content, reasoning = partial
        _LOCAL.partial = None
        if getattr(_LOCAL, "loop_cut", None):  # cut early as a loop (its loop_ratio at the cut)
            rec["loop_cut"] = _LOCAL.loop_cut
            _LOCAL.loop_cut = None
        if getattr(_LOCAL, "t_first_token", None):
            rec["t_first_token"] = _LOCAL.t_first_token
            _LOCAL.t_first_token = None
        rec["chars"], rec["reasoning_chars"] = len(content), len(reasoning)
        tail = (reasoning + content)[-LOOP_TAIL:].encode()
        if len(tail) >= 2000:  # zlib ratio of the reply's end: a normal answer ~0.25-0.4, a loop < 0.1
            rec["loop_ratio"] = round(len(zlib.compress(tail)) / len(tail), 4)
        if rec.get("finish_reason") in ("length", TIMEOUT_REPLY):  # cut: keep its end, to tell a loop from work
            rec["tail_content"], rec["tail_reasoning"] = content[-KEEP_TAIL:], reasoning[-KEEP_TAIL:]

    proxy.Proxy.tally = noted


def _container_cleanup():
    """Stop a harness's container when its command stops: QualityBench kills the `podman run` client at the deadline
    (or on a signal) but the container itself keeps going (LiveCodeBench's generator retried a dead server for a day,
    mounted read-write in the run's folder). Each container gets a name (qb-<pid>-<n>); when the command that started
    it returns or raises, `<runtime> rm -f` removes it. SIGTERM unwinds like Ctrl-C so that runs; atexit takes the rest."""
    import atexit
    import itertools
    import signal
    import subprocess

    from qualitybench import context, envs

    started, count = set(), itertools.count()
    container_run = envs.container_run

    def named(net, mounts, extra=None):
        cmd = container_run(net, mounts, extra)
        name = f"qb-{os.getpid()}-{next(count)}"
        started.add(name)
        return cmd[:2] + ["--name", name] + cmd[2:]

    envs.container_run = named

    def remove(names):
        names = [n for n in names if n in started]
        if not names:
            return
        try:
            subprocess.run([envs.container(), "rm", "-f", "-t", "5", *names], capture_output=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            pass
        started.difference_update(names)

    sh = context.Ctx.sh

    def cleaned(self, cmd, *args, **kwargs):
        argv = [str(a) for a in cmd]
        names = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == "--name"]
        try:
            return sh(self, cmd, *args, **kwargs)
        finally:
            remove(names)  # exited with --rm already: a no-op

    context.Ctx.sh = cleaned
    atexit.register(lambda: remove(list(started)))

    def on_term(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)


def main(argv):
    _count_tokens()
    _reply_timings()  # after _count_tokens: its tally wraps that one
    window = int(os.environ.get("QB_REPLY_WINDOW") or 3600)
    _lcb_timeout(window + 600)
    _reply_window(window)
    _container_cleanup()  # after _reply_window: it wraps the Ctx.sh that one patched
    if image := os.environ.get("QB_EVALPLUS_IMAGE"):
        import qualitybench.envs as envs

        envs.EVALPLUS_IMAGE = image
    if floor := int(os.environ.get("QB_MAX_TOKENS_FLOOR") or 0):
        _max_tokens_floor(floor)
    if os.environ.get("QB_PREFLIGHT_RETRY"):
        _preflight_retry()
    if os.environ.get("QB_RESTREAM", "0") != "0":  # opt-in until checked on a real server
        _restream()  # last: it sees the request as the other rewrites leave it
    from qualitybench.cli import main as qb_main

    return qb_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
