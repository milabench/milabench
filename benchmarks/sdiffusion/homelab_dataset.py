#!/usr/bin/env python
"""Build the homelab fine-tuning set from the suite's own generations (CPU only, Pillow + ffprobe).

Only outputs of Apache-2.0 models are taken (Z-Image Turbo, FLUX.1 [schnell], Wan 2.2 TI2V 5B): their outputs
carry no use restriction. FLUX.1 [dev] / Krea outputs are left out (non-commercial / custom licenses).
Captions are the English prompts the suite generated them from (prompts.json of the vLLM-Omni benches, the
ComfyUI workflow embedded in each PNG / MP4).

Layout (what benchmate.diffusion.materialize_dataset produces, so the sdiffusion packs take it as is with
``--dataset <name>``; its ``.complete`` marker stops them from downloading the default dataset):

    <cache>/sdiffusion/datasets/<name>/train/images/NNNNN.jpg + NNNNN.txt + metadata.jsonl
    <cache>/sdiffusion/datasets/<name>/val/images/...
    <cache>/sdiffusion/datasets/<name>/videos/NNNNN.mp4 + NNNNN.txt          (Wan video LoRA)
    <cache>/sdiffusion/datasets/<name>/PROVENANCE.json

metadata.jsonl (file_name, text) makes train/images a Hugging Face imagefolder: benchmarks/diffusion (the SD2
full fine-tune) loads it with ``--dataset <that folder>``.

    python benchmarks/sdiffusion/homelab_dataset.py --runs $MILABENCH_BASE/runs --cache $MILABENCH_BASE/cache
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

# bench output folder prefix -> (model, license)
SOURCES = {
    "homelab-comfyui-z-image-turbo": ("Tongyi-MAI/Z-Image-Turbo", "apache-2.0"),
    "homelab-omni-z-image-turbo": ("Tongyi-MAI/Z-Image-Turbo", "apache-2.0"),
    "homelab-comfyui-flux-schnell": ("black-forest-labs/FLUX.1-schnell", "apache-2.0"),
    "homelab-omni-flux-schnell-fp8": ("black-forest-labs/FLUX.1-schnell", "apache-2.0"),
}
VIDEO_SOURCES = {
    "homelab-comfyui-wan22-5b-ti2v": ("Wan-AI/Wan2.2-TI2V-5B", "apache-2.0"),
}
TEXT_NODES = ("CLIPTextEncode", "CLIPTextEncodeFlux", "TextEncodeQwenImageEdit")


def png_prompt(path):
    """Positive prompt of a ComfyUI PNG (workflow JSON in its tEXt 'prompt' chunk)."""
    data = open(path, "rb").read()
    i = 8
    while i < len(data):
        (n,) = struct.unpack(">I", data[i:i + 4])
        kind = data[i + 4:i + 8]
        if kind == b"tEXt":
            key, value = data[i + 8:i + 8 + n].split(b"\0", 1)
            if key == b"prompt":
                return workflow_prompt(json.loads(value))
        if kind == b"IDAT":
            break
        i += 12 + n
    return None


def workflow_prompt(workflow):
    texts = [node["inputs"].get("text") for node in workflow.values()
             if node.get("class_type") in TEXT_NODES and isinstance(node["inputs"].get("text"), str)]
    # the positive prompt is the first non-empty text that is not the long negative list
    texts = [t for t in texts if t.strip() and not t.startswith(("garish", "blurry", "low quality"))]
    return texts[0] if texts else None


def video_prompt(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
                             capture_output=True, text=True, check=True).stdout
        tags = json.loads(out).get("format", {}).get("tags", {})
        return workflow_prompt(json.loads(tags["prompt"])) if "prompt" in tags else None
    except Exception:
        return None


def collect(runs, sources, pattern, caption_of):
    seen, items = set(), []
    for folder in sorted(glob.glob(os.path.join(runs, "*", "*.outputs"))):
        bench = os.path.basename(folder)[: -len(".outputs")]
        if bench not in sources:
            continue
        prompts = {}
        if os.path.exists(os.path.join(folder, "prompts.json")):
            prompts = json.load(open(os.path.join(folder, "prompts.json")))
        for path in sorted(glob.glob(os.path.join(folder, pattern))):
            digest = hashlib.md5(open(path, "rb").read()).hexdigest()
            if digest in seen:
                continue
            caption = prompts.get(os.path.basename(path)) or caption_of(path)
            if not caption:
                continue
            seen.add(digest)
            model, license_ = sources[bench]
            items.append({"source": path, "md5": digest, "caption": caption, "model": model, "license": license_})
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(os.environ.get("MILABENCH_BASE", "/opt/milabench"), "runs"))
    parser.add_argument("--cache", default=os.path.join(os.environ.get("MILABENCH_BASE", "/opt/milabench"), "cache"))
    parser.add_argument("--name", default="homelab-suite-apache")
    parser.add_argument("--val", type=int, default=2, help="images held out as val")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dest = Path(args.cache) / "sdiffusion" / "datasets" / args.name
    if (dest / ".complete").exists() and not args.force:
        print(f"{dest} already built (--force to rebuild)")
        return
    if dest.exists():
        shutil.rmtree(dest)

    from PIL import Image

    images = collect(args.runs, SOURCES, "*.png", png_prompt)
    videos = collect(args.runs, VIDEO_SOURCES, "*.mp4", video_prompt)
    if len(images) <= args.val:
        raise SystemExit(f"only {len(images)} images found under {args.runs}")

    # hold out one image per prompt at most, from the end (the suite repeats 4 prompts across models)
    train, val = images[: -args.val], images[-args.val:]
    for split, items in (("train", train), ("val", val)):
        folder = dest / split / "images"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "metadata.jsonl", "w") as meta:
            for i, item in enumerate(items):
                name = f"{i:05d}"
                Image.open(item["source"]).convert("RGB").save(folder / f"{name}.jpg", quality=95)
                (folder / f"{name}.txt").write_text(item["caption"])
                meta.write(json.dumps({"file_name": f"{name}.jpg", "text": item["caption"]}) + "\n")
                item["file"] = f"{split}/images/{name}.jpg"

    vfolder = dest / "videos"
    vfolder.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(videos):
        shutil.copyfile(item["source"], vfolder / f"{i:05d}.mp4")
        (vfolder / f"{i:05d}.txt").write_text(item["caption"])
        item["file"] = f"videos/{i:05d}.mp4"

    (dest / "PROVENANCE.json").write_text(json.dumps({
        "note": "generated by the homelab suite from Apache-2.0 models; captions are the generation prompts",
        "train": train, "val": val, "videos": videos,
    }, indent=1))
    (dest / ".complete").write_text(json.dumps({
        "dataset_id": f"local:{args.name}", "train": len(train), "val": len(val), "videos": len(videos),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }))
    print(f"{dest}: {len(train)} train / {len(val)} val images, {len(videos)} videos, "
          f"{len({i['caption'] for i in images})} distinct captions")


if __name__ == "__main__":
    main()
