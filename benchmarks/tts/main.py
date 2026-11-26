#!/usr/bin/env python
"""Text-to-speech / voice-assistant client: milabench runs ``vllm serve <model> --omni`` (backend vllm-omni,
milabench.commands.serving), this script sends the pinned English texts (texts.py) and times the audio.

Modes:
  speech  POST /v1/audio/speech (TTS models). Streaming (default): raw PCM over HTTP, the first chunk gives
          the time to first audio. --no-stream: one WAV per request.
  chat    POST /v1/chat/completions with modalities [text, audio] (omni models: the voice-assistant turn).
          SSE stream: text deltas and WAV chunks. --audio-input self: each question is first read aloud by the
          model itself (untimed) and sent as the user's audio turn, so the timed turn is speech in -> speech out.

One step = --concurrency requests in flight. The step rate is the aggregate audio throughput: seconds of
audio produced per second of wall time (x real time). Per request: ttfa_s (time to first audio chunk),
gen_seconds, audio_seconds, rtf_x = audio_seconds / gen_seconds (> 1: faster than real time).

--wer: after the timed steps, the first take of every text is transcribed with Whisper (--asr-model, on the
GPU, server idle) and the corpus word error rate against the input text is recorded (chat: against the
model's own text answer, i.e. whether the voice says what the model wrote).

Client passes (client.passes in the config, ``--pass <name>``): the pass' argv overlays the shared client argv.
"""
import base64
import io
import itertools
import json
import os
import re
import sys
import threading
import time
import urllib.request
from argparse import ArgumentParser, BooleanOptionalAction
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import texts  # noqa: E402


# ---------------------------------------------------------------------------
# Config: shared client argv + the pass overlay
# ---------------------------------------------------------------------------


def _milabench_config():
    try:
        return json.loads(os.environ.get("MILABENCH_CONFIG", "{}"))
    except ValueError:
        return {}


def _options(overlay):
    argv = []
    for key, value in (overlay or {}).items():
        if value is True:
            argv.append(key)
        elif value is not False and value is not None:
            argv += [key, str(value)]
    return argv


def pass_overlay(name):
    """argv of the client pass *name* (the arch variant's passes replace the shared ones)."""
    if not name:
        return []
    cfg = _milabench_config()
    arch = cfg.get("system", {}).get("arch", "cuda")
    variant = ((cfg.get("variants") or {}).get(arch) or {}).get("client") or {}
    passes = variant.get("passes") or (cfg.get("client") or {}).get("passes") or []
    for spec in passes:
        if spec["name"] == name:
            return _options(spec.get("argv"))
    raise SystemExit(f"no client pass named {name!r} in {[spec['name'] for spec in passes]}")


def arguments(argv):
    parser = ArgumentParser()
    parser.add_argument("--pass", dest="pass_name", default=None)
    parser.add_argument("--mode", choices=["speech", "chat"], default="speech")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--texts", choices=sorted(texts.SETS), default="short")
    parser.add_argument("--concurrency", type=int, default=1, help="requests in flight per step")
    parser.add_argument("--warmup", type=int, default=1, help="untimed steps first (graphs, caches)")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = voir decides)")
    parser.add_argument("--stream", action=BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=600)
    # speech request fields (None: not sent, the model's default)
    parser.add_argument("--voice", default=None)
    parser.add_argument("--instructions", default=None, help="voice design / style prompt")
    parser.add_argument("--task-type", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--sample-rate", type=int, default=None, help="resample on the server (default: native)")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--extra-params", default=None, help="JSON object: model-specific sampling parameters")
    # chat (omni) fields
    parser.add_argument("--audio-input", choices=["none", "self"], default="none")
    parser.add_argument("--max-tokens", type=int, default=256)
    # round-trip intelligibility
    parser.add_argument("--wer", action=BooleanOptionalAction, default=False)
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    # the server side's, ignored here
    parser.add_argument("--ready-check-timeout-sec", type=float, default=None)
    args, _ = parser.parse_known_args(argv)
    if args.pass_name:
        args, _ = parser.parse_known_args([*argv, *pass_overlay(args.pass_name)])
    return args


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


def _post(args, path, payload):
    req = urllib.request.Request(
        f"http://{args.host}:{args.port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=args.timeout)


def _wav_to_array(data):
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr


def speech_request(args, text, index):
    payload = {"input": text, "response_format": "pcm" if args.stream else "wav", "seed": args.seed + index}
    for key, value in (("voice", args.voice), ("instructions", args.instructions), ("task_type", args.task_type),
                       ("language", args.language), ("sample_rate", args.sample_rate)):
        if value is not None:
            payload[key] = value
    if args.extra_params:
        payload["extra_params"] = json.loads(args.extra_params)
    if args.stream:
        payload.update(stream=True, stream_format="audio")

    start = time.perf_counter()
    first = None
    chunks = []
    with _post(args, "/v1/audio/speech", payload) as resp:
        sr = int(resp.headers.get("X-Audio-Sample-Rate") or args.sample_rate or 24000)
        channels = int(resp.headers.get("X-Audio-Channels") or 1)
        while True:
            chunk = resp.read1(65536) if args.stream else resp.read()
            if not chunk:
                break
            if first is None:
                first = time.perf_counter() - start
            chunks.append(chunk)
            if not args.stream:
                break
    total = time.perf_counter() - start
    data = b"".join(chunks)
    if args.stream:
        pcm = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm[: len(pcm) // channels * channels].reshape(-1, channels).mean(axis=1)
        audio = pcm
    else:
        audio, sr = _wav_to_array(data)
    return {"ttfa": first if args.stream else total, "total": total, "audio": audio, "sr": sr,
            "reference": text}


def _audio_turn(audio_b64):
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}}


def chat_request(args, question, index, audio_b64=None):
    user = [_audio_turn(audio_b64)] if audio_b64 else question
    payload = {
        "messages": [{"role": "system", "content": texts.ASSISTANT_SYSTEM}, {"role": "user", "content": user}],
        "modalities": ["text", "audio"], "audio": {"format": "wav"},
        "stream": True, "max_tokens": args.max_tokens, "temperature": 0.0, "seed": args.seed + index,
    }
    if args.voice:
        payload["audio"]["voice"] = args.voice
    start = time.perf_counter()
    first_text = first_audio = None
    words, pieces, sr = [], [], 24000
    with _post(args, "/v1/chat/completions", payload) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[5:]).get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if not content:
                    continue
                if choice.get("audio_metadata") is not None:  # base64 WAV chunk
                    if first_audio is None:
                        first_audio = time.perf_counter() - start
                    audio, sr = _wav_to_array(base64.b64decode(content))
                    pieces.append(audio)
                else:
                    if first_text is None:
                        first_text = time.perf_counter() - start
                    words.append(content)
    total = time.perf_counter() - start
    audio = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return {"ttfa": first_audio if first_audio is not None else total, "ttft": first_text, "total": total,
            "audio": audio, "sr": sr, "reference": "".join(words)}


def read_aloud(args, question):
    """The spoken user turn: the omni model reads the question aloud (non-streamed, untimed)."""
    payload = {
        "messages": [{"role": "user", "content": texts.READ_ALOUD + question}],
        "modalities": ["audio"], "audio": {"format": "wav"}, "stream": False, "temperature": 0.0,
        "max_tokens": 128,
    }
    with _post(args, "/v1/chat/completions", payload) as resp:
        body = json.load(resp)
    for choice in body.get("choices") or []:
        audio = (choice.get("message") or {}).get("audio") or {}
        if audio.get("data"):
            return audio["data"]
    raise RuntimeError(f"no audio in the read-aloud answer for {question!r}")


# ---------------------------------------------------------------------------
# Outputs, steps
# ---------------------------------------------------------------------------

KEEP_OUTPUTS = int(os.environ.get("MILABENCH_KEEP_OUTPUTS", "4"))
_kept = {"count": 0, "lock": threading.Lock()}


def keep_output(args, result):
    """The first few clips next to the run's results, <bench>.outputs/, with their texts (texts.json)."""
    run_dir = os.environ.get("MILABENCH_DIR_RUN")
    if not run_dir or not len(result["audio"]):
        return
    with _kept["lock"]:
        if _kept["count"] >= KEEP_OUTPUTS:
            return
        _kept["count"] += 1
        index = _kept["count"]
    import soundfile as sf

    name = _milabench_config().get("name") or "tts"
    dest = os.path.join(run_dir, f"{name}.outputs")
    os.makedirs(dest, exist_ok=True)
    filename = f"tts_{index:03d}.wav"
    sf.write(os.path.join(dest, filename), result["audio"], result["sr"])
    with _kept["lock"]:
        index_file = os.path.join(dest, "texts.json")
        known = json.load(open(index_file)) if os.path.exists(index_file) else {}
        known[filename] = result["reference"]
        json.dump(known, open(index_file, "w"), indent=1)


class Workload:
    def __init__(self, args):
        self.args = args
        self.items = texts.SETS["questions" if args.mode == "chat" else args.texts]
        self.index = 0
        self.lock = threading.Lock()
        self.spoken = {}
        self.takes = {}  # item -> first result, for --wer

    def prepare(self):
        if self.args.mode == "chat" and self.args.audio_input == "self":
            for question in self.items:
                self.spoken[question] = read_aloud(self.args, question)
                print(f"spoken turn ready: {question}", flush=True)

    def request(self, timed):
        with self.lock:
            index = self.index
            self.index += 1
        item = self.items[index % len(self.items)]
        if self.args.mode == "chat":
            result = chat_request(self.args, item, index, self.spoken.get(item))
        else:
            result = speech_request(self.args, item, index)
        result["audio_seconds"] = len(result["audio"]) / result["sr"]
        if timed:
            try:
                keep_output(self.args, result)
            except (OSError, ValueError) as exc:
                print(f"could not keep the output: {exc}", flush=True)
            if self.args.wer:
                with self.lock:
                    self.takes.setdefault(item, result)
        return result

    def step(self, pool, timed=True):
        jobs = [pool.submit(self.request, timed) for _ in range(self.args.concurrency)]
        return [job.result() for job in jobs]


# ---------------------------------------------------------------------------
# Round-trip intelligibility
# ---------------------------------------------------------------------------


def _edit_distance(ref, hyp):
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def word_error_rate(takes, asr_model):
    """Corpus WER (sum of word edits / sum of reference words) of Whisper's transcripts of the takes."""
    import torch
    import torchaudio.functional as AF
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model=asr_model, torch_dtype=torch.float16, device="cuda:0")

    def normalize(text):
        try:
            return asr.tokenizer.normalize(text)
        except Exception:  # noqa: BLE001 - a plain fallback is good enough
            return re.sub(r"[^a-z0-9' ]+", " ", text.lower())

    edits = words = 0
    for result in takes:
        if not len(result["audio"]) or not result["reference"].strip():
            continue
        audio = AF.resample(torch.from_numpy(result["audio"]), result["sr"], 16000).numpy()
        hyp = asr({"raw": audio, "sampling_rate": 16000}, chunk_length_s=30,
                  generate_kwargs={"language": "en", "task": "transcribe"})["text"]
        ref_words, hyp_words = normalize(result["reference"]).split(), normalize(hyp).split()
        edits += _edit_distance(ref_words, hyp_words)
        words += len(ref_words)
        print(f"wer item: {_edit_distance(ref_words, hyp_words)}/{len(ref_words)} | {hyp.strip()}", flush=True)
    del asr
    torch.cuda.empty_cache()
    return edits / max(words, 1), words


# ---------------------------------------------------------------------------


def prepare_voir():
    import torchcompat.core as accelerator
    from benchmate.monitor import bench_monitor
    from benchmate.observer import BenchObserver
    from benchmate.toggles import get_observation_count

    observer = BenchObserver(
        accelerator.Event,
        earlystop=get_observation_count(30),
        batch_size_fn=lambda x: 1,
        raise_stop_program=False,
        stdout=True,
    )
    return observer, bench_monitor


def main(argv=None):
    args = arguments(sys.argv[1:] if argv is None else argv)
    work = Workload(args)
    work.prepare()

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        for _ in range(args.warmup):
            for r in work.step(pool, timed=False):
                print(f"warmup: {r['audio_seconds']:.2f}s of audio in {r['total']:.2f}s", flush=True)

        observer, monitor = prepare_voir()
        dataset = observer.loader(itertools.repeat(None), custom_step=True)
        with monitor():
            for i, _ in enumerate(dataset):
                t0 = time.perf_counter()
                results = work.step(pool)
                wall = time.perf_counter() - t0
                audio_total = sum(r["audio_seconds"] for r in results)
                for r in results:
                    metric = dict(ttfa_s=r["ttfa"], gen_seconds=r["total"], audio_seconds=r["audio_seconds"],
                                  rtf_x=r["audio_seconds"] / r["total"] if r["total"] else 0.0,
                                  concurrency=args.concurrency, task=args.mode)
                    if r.get("ttft") is not None:
                        metric["ttft_s"] = r["ttft"]
                    observer.record_metric(**metric)
                observer.record_metric(step_wall_seconds=wall, step_requests_per_s=len(results) / wall,
                                       step_audio_x_realtime=audio_total / wall, concurrency=args.concurrency)
                # the step's rate: seconds of audio per second of wall time
                dataset.step(audio_total)
                if args.max_steps and i + 1 >= args.max_steps:
                    break

    if args.wer and work.takes:
        wer, words = word_error_rate(list(work.takes.values()), args.asr_model)
        observer.record_metric(wer=wer, wer_words=words, wer_items=len(work.takes), asr_model=args.asr_model)
        print(f"round-trip WER: {wer:.4f} over {words} words", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
