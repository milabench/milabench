#!/usr/bin/env python
"""Download the served model (server argv positional) and, for the round-trip WER passes, the Whisper model
(client --asr-model) into the HF cache: the runs are offline.

argv: <server argv> -- <client argv> (benchfile.build_prepare_plan)."""
import sys

from benchmate.hugginface import download_hf_model


def main(argv):
    server, client = (argv[: argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    # the model is the first positional value with a "/" (keep it first in server.argv: a flag value such as a
    # deploy path could match otherwise; paths and JSON overrides are skipped)
    model = next((arg for arg in server if not arg.startswith(("-", "{")) and "/" in arg
                  and not arg.endswith((".yaml", ".json"))), None)
    if model is None:
        raise SystemExit("prepare: no model in the server argv")
    download_hf_model(model)
    print(f"prepared {model}")

    # Kokoro's English G2P (misaki, vllm-omni's [kokoro] extra) loads spaCy's en_core_web_sm and would download
    # it on first use, during the offline run: fetch it here
    if "kokoro" in model.lower():
        import spacy.cli
        import spacy.util

        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")
        print("prepared spaCy en_core_web_sm (Kokoro G2P)")

    # the WER passes transcribe with Whisper: only when a pass asks for it (--wer in the config)
    from main import arguments, _milabench_config

    cfg = _milabench_config()
    passes = (cfg.get("client") or {}).get("passes") or []
    if any((spec.get("argv") or {}).get("--wer") for spec in passes) or "--wer" in client:
        asr = arguments(client).asr_model
        download_hf_model(asr)
        print(f"prepared {asr}")


if __name__ == "__main__":
    main(sys.argv[1:])
