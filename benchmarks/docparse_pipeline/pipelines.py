"""Classic (non-LLM) page -> text pipelines, the baseline of the docparse VLM benches.

    tesseract   Tesseract 5.5 through tesserocr (its aarch64 wheel bundles libtesseract + leptonica), LSTM models
                from tessdata_best (eng + fra, pinned), page segmentation auto. CPU only. Plain text out.
    rapidocr    PaddleOCR PP-OCRv6 (det + rec, ``medium``, multilingual rec) as ONNX models through RapidOCR and
                onnxruntime (no paddlepaddle: PyPI has no paddlepaddle-gpu wheel for aarch64). CPU by default,
                ``--device cuda``: onnxruntime-gpu's CUDA execution provider. Lines ordered by RapidOCR's
                ``to_markdown`` (top to bottom, left to right: no column detection).
    docling     docling's standard PDF pipeline: layout model (Heron) + TableFormer on torch (``--device cuda``)
                and OCR (RapidOCR PP-OCR, or tesseract) where the page has no text layer. A PDF with a text layer
                is read from it (docling's normal use); ``--docling-force-ocr`` OCRs every page instead. Markdown
                out (tables as Markdown tables).

Each engine is built once per worker process (``--concurrency`` workers, each its own copy of the models).
"""

from __future__ import annotations

import os
import time

# tessdata_best at its latest tag (the LSTM models Tesseract 4.1+ / 5.x read)
TESSDATA_REF = "4.1.0"
TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_best/raw/{ref}/{lang}.traineddata"
TESS_LANGS = ("eng", "fra", "osd")


def tessdata_dir() -> str:
    base = os.environ.get("MILABENCH_DIR_DATA") or os.path.join(os.path.expanduser("~"), ".cache", "milabench")
    return os.path.join(base, "docparse", "tessdata_best", TESSDATA_REF)


def prepare_tessdata() -> str:
    import urllib.request

    out = tessdata_dir()
    os.makedirs(out, exist_ok=True)
    for lang in TESS_LANGS:
        path = os.path.join(out, f"{lang}.traineddata")
        if not os.path.exists(path):
            urllib.request.urlretrieve(TESSDATA_URL.format(ref=TESSDATA_REF, lang=lang), path + ".part")
            os.rename(path + ".part", path)
    return out


class Tesseract:
    needs = "image"

    def __init__(self, opts: dict):
        import tesserocr

        self.api = tesserocr.PyTessBaseAPI(path=tessdata_dir() + "/", lang=opts.get("langs", "eng+fra"),
                                           psm=tesserocr.PSM.AUTO)

    def __call__(self, image, path: str) -> str:
        self.api.SetImage(image)
        return self.api.GetUTF8Text()


class RapidOcr:
    needs = "image"

    def __init__(self, opts: dict):
        from rapidocr import RapidOCR

        params = {"Det.ocr_version": "PP-OCRv6", "Rec.ocr_version": "PP-OCRv6",
                  "Det.model_type": opts.get("model_type", "medium"), "Rec.model_type": opts.get("model_type", "medium"),
                  "EngineConfig.onnxruntime.use_cuda": opts.get("device") == "cuda"}
        self.engine = RapidOCR(params=params)

    def __call__(self, image, path: str) -> str:
        import numpy as np

        result = self.engine(np.asarray(image))
        return result.to_markdown() if result.txts else ""


class Docling:
    needs = "file"

    def __init__(self, opts: dict):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TesseractOcrOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        except ImportError:  # older layout
            from docling.datamodel.pipeline_options import AcceleratorDevice, AcceleratorOptions

        force = bool(opts.get("force_ocr"))
        ocr = (TesseractOcrOptions(lang=["eng", "fra"], force_full_page_ocr=force) if opts.get("ocr") == "tesseract"
               else RapidOcrOptions(force_full_page_ocr=force))
        if opts.get("ocr") == "tesseract":
            os.environ.setdefault("TESSDATA_PREFIX", tessdata_dir() + "/")
        device = AcceleratorDevice.CUDA if opts.get("device") == "cuda" else AcceleratorDevice.CPU
        pipeline = PdfPipelineOptions(do_ocr=True, do_table_structure=True, ocr_options=ocr,
                                      accelerator_options=AcceleratorOptions(device=device, num_threads=4))
        self.converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline)})

    def __call__(self, image, path: str) -> str:
        return self.converter.convert(path).document.export_to_markdown()


ENGINES = {"tesseract": Tesseract, "rapidocr": RapidOcr, "docling": Docling}

# ---------------------------------------------------------------------------
# Worker processes: one engine each
# ---------------------------------------------------------------------------

_engine = None


def _init(name: str, opts: dict):
    global _engine
    _engine = ENGINES[name](opts)


def _one(job):
    """job: (path, image or None) -> {reply, error, e2el_s, ...}"""
    path, image = job
    start = time.perf_counter()
    try:
        reply, error = _engine(image, path), None
    except Exception as exc:  # noqa: BLE001 - one failed page is one empty page
        reply, error = "", f"{type(exc).__name__}: {exc}"
    return {"reply": reply, "error": error, "e2el_s": time.perf_counter() - start, "ttft_s": None,
            "prompt_tokens": None, "completion_tokens": None, "finish_reason": None}


def warm(name: str, opts: dict) -> None:
    """prepare: build the engine once (downloads its models: RapidOCR's ONNX files, docling's layout / table
    models from Hugging Face)."""
    ENGINES[name](opts)
