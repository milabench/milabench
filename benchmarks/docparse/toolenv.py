"""Pinned scorer and vendor-SDK venvs of the docparse benches (their dependencies clash with vLLM's / docling's).

Same layout and installer as benchmate.benchserve.tools ($MILABENCH_DIR_DATA/benchserve-tools/<name>, uv),
with this pack's own pins. Installed by prepare.py (network), used offline by the client passes:

    docparse-score   olmocr[bench] (olmOCR-bench scorer + KaTeX rendering through playwright's chromium),
                     omni-parse-bench (scorer), docling-core (DocTags -> Markdown), markdownify; runs convert.py
    omnidocbench     the official OmniDocBench evaluator at a pinned commit (Python 3.11: its pins need it);
                     CDM (formula rendering through TeX Live + ImageMagick) is not installed: formulas are
                     scored by edit distance
    docparse-sdk     vendor clients that drive the served model themselves: mineru-vl-utils (MinerU2.5 two-step
                     extraction), paddleocr (PaddleOCR-VL: PP-DocLayoutV3 layout on the CPU + region requests to
                     the server), and chandra-ocr without its dependencies (its prompt text only)
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
import urllib.request

from benchmate.benchserve import tools

OMNIDOCBENCH_COMMIT = "7ddb8df411ad9bd2fff139acafe0ed16aa1694fe"

ENVS = {
    "docparse-score": {
        "python": "3.12",
        "specs": ["olmocr[bench]==0.4.27", "omni-parse-bench==1.5.0", "docling-core==2.107.1",
                  "markdownify==1.2.3", "playwright==1.65.2"],
        "post": [["-m", "playwright", "install", "chromium"]],
    },
    "omnidocbench": {
        "python": "3.11",
        "source": f"https://codeload.github.com/opendatalab/OmniDocBench/tar.gz/{OMNIDOCBENCH_COMMIT}",
        "specs": [],  # the checkout itself (pyproject: its pinned dependencies)
        # its lxml==4.9.1 has no linux-aarch64 wheel for Python 3.11 (a source build needs libxml2-dev, absent):
        # 4.9.4, the same 4.9 line, has one
        "overrides": ["lxml==4.9.4"],
        # its text metrics load evaluate's bleu / meteor and NLTK data from the network: fetched here, read
        # offline at run time (omnidocbench_env)
        "post": [["-c", "import nltk, evaluate; [nltk.download(p, quiet=True) for p in ('wordnet', 'punkt', "
                        "'punkt_tab', 'omw-1.4')]; evaluate.load('bleu'); evaluate.load('meteor')"]],
        "post_env": "omnidocbench",
    },
    "docparse-sdk": {
        "python": "3.12",
        "specs": ["mineru-vl-utils==2.0.9", "paddleocr[doc-parser]==3.7.0",
                  # aarch64: PyPI's last paddlepaddle wheel for linux-aarch64 is 3.2.2 (CPU only: the layout
                  # model runs on the CPU, the VLM on the server)
                  "paddlepaddle==3.2.2", "pypdfium2==5.15.2"],
        "nodeps": ["chandra-ocr==0.2.0"],
    },
}


def venv(name: str) -> str:
    return tools.venv(name)


def python(name: str) -> str:
    return tools.python(name)


def source_dir(name: str) -> str:
    return os.path.join(venv(name), "src")


def omnidocbench_env(offline: bool = True) -> dict:
    """Caches of the OmniDocBench evaluator inside its venv; offline once prepared."""
    base = venv("omnidocbench")
    env = {"HF_HOME": os.path.join(base, "hf"), "NLTK_DATA": os.path.join(base, "nltk_data")}
    env["XDG_CACHE_HOME"] = os.path.join(base, "cache")
    if offline:
        env.update({"HF_EVALUATE_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    return env


def ensure(name: str) -> str:
    env = ENVS[name]
    marker = os.path.join(venv(name), ".docparse-spec")
    spec = repr(env)
    if os.path.exists(python(name)) and os.path.exists(marker) and open(marker).read() == spec:
        return python(name)
    os.makedirs(tools.root(), exist_ok=True)
    uv = tools._uv()
    subprocess.run([*uv, "venv", "--allow-existing", "-p", env["python"], venv(name)], check=True)
    specs = list(env["specs"])
    if env.get("source"):
        src = source_dir(name)
        if not os.path.isdir(src):
            with tempfile.TemporaryDirectory() as tmp:
                archive = os.path.join(tmp, "src.tar.gz")
                urllib.request.urlretrieve(env["source"], archive)
                with tarfile.open(archive) as tar:
                    tar.extractall(tmp, filter="data")
                top = next(d for d in os.listdir(tmp) if os.path.isdir(os.path.join(tmp, d)))
                os.rename(os.path.join(tmp, top), src)
        specs.append(src)
    if specs:
        extra = []
        if env.get("overrides"):
            overrides = os.path.join(venv(name), "overrides.txt")
            with open(overrides, "w") as f:
                f.write("\n".join(env["overrides"]) + "\n")
            extra = ["--override", overrides]
        subprocess.run([*uv, "pip", "install", "--python", python(name), *extra, *specs], check=True)
    if env.get("nodeps"):
        subprocess.run([*uv, "pip", "install", "--python", python(name), "--no-deps", *env["nodeps"]], check=True)
    post_env = {**os.environ, **omnidocbench_env(offline=False)} if env.get("post_env") == "omnidocbench" else None
    for args in env.get("post", []):
        subprocess.run([python(name), *args], check=True, env=post_env)
    with open(marker, "w") as f:
        f.write(spec)
    return python(name)
