"""A model's raw reply -> the page as Markdown, the input of every scorer.

Runs in the scorer venv (toolenv.py ``docparse-score``: docling-core for DocTags, markdownify for HTML), never
in the serving venv, so the client keeps the raw reply and conversion bugs can be fixed without a GPU run:

    <python of docparse-score> convert.py <answers.jsonl> <out_dir>      # one <out_dir>/<safe id>.md per page

Formats (``docparse.output`` of a config entry):

    markdown      the reply is Markdown already (a ```markdown fence around the page is removed)
    olmocr        Markdown after a YAML front matter block (olmOCR's PageResponse): the front matter is dropped
    dots_json     dots.ocr / dots.mocr layout JSON, a list of {bbox, category, text} in reading order: headers,
                  footers and pictures dropped, titles and section headers as headings, formulas as $$...$$,
                  tables kept as HTML
    doctags       Granite-Docling's DocTags, exported by docling-core
    html          HTML (chandra's layout divs, ...): converted with markdownify, tables kept as HTML
    deepseek      DeepSeek-OCR grounding output: <|ref|>label<|/ref|><|det|>[[boxes]]<|/det|> markers removed,
                  image blocks dropped
    nemotron      Nemotron Parse: <x_..><y_..> box tokens and <class_..> tags removed, page headers and footers
                  dropped
"""

from __future__ import annotations

import json
import os
import re
import sys


def _unfence(text: str) -> str:
    m = re.match(r"^\s*```(?:markdown|md|html)?\s*\n(.*?)(?:\n```\s*)?$", text, re.S)
    return m.group(1) if m else text


def markdown(text: str) -> str:
    return _unfence(text).strip()


def olmocr(text: str) -> str:
    text = _unfence(text)
    m = re.match(r"^\s*---\s*\n.*?\n---\s*\n?(.*)$", text, re.S)
    return (m.group(1) if m else text).strip()


def dots_json(text: str) -> str:
    text = _unfence(text).strip()
    try:
        blocks = json.loads(text)
    except json.JSONDecodeError:
        # a reply cut at max_tokens: keep the complete blocks
        cut = text.rfind("},")
        try:
            blocks = json.loads(text[: cut + 1] + "]") if cut > 0 else []
        except json.JSONDecodeError:
            return text
    if isinstance(blocks, dict):
        blocks = blocks.get("layout") or blocks.get("blocks") or [blocks]
    out = []
    for b in blocks:
        cat, body = b.get("category", "Text"), (b.get("text") or "").strip()
        if cat in ("Page-header", "Page-footer", "Picture") or not body:
            continue
        if cat == "Title":
            out.append(f"# {body}")
        elif cat == "Section-header":
            out.append(f"## {body}")
        elif cat == "Formula":
            out.append(body if body.startswith("$$") else f"$$\n{body.strip('$ ')}\n$$")
        else:
            out.append(body)
    return "\n\n".join(out)


def doctags(text: str) -> str:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument

    pairs = DocTagsDocument.from_doctags_and_image_pairs([text], [None])
    doc = DoclingDocument.load_from_doctags(pairs, document_name="page")
    return doc.export_to_markdown()


def html(text: str) -> str:
    from markdownify import MarkdownConverter

    class KeepTables(MarkdownConverter):
        def convert_table(self, el, text, *args, **kwargs):  # tables stay HTML: the scorers read HTML tables
            return "\n\n" + str(el) + "\n\n"

        def convert_math(self, el, text, *args, **kwargs):
            body = el.get_text()
            return f"\n\n$$\n{body}\n$$\n\n" if el.get("display") else f"${body}$"

    text = _unfence(text)
    text = re.sub(r'<div[^>]*data-label="(?:Page-Header|Page-Footer|Image|Figure)"[^>]*>.*?</div>', "", text,
                  flags=re.S | re.I)
    return KeepTables(heading_style="ATX").convert(text).strip()


def deepseek(text: str) -> str:
    text = re.sub(r"<\|ref\|>image<\|/ref\|><\|det\|>.*?<\|/det\|>\s*", "", text, flags=re.S)
    text = re.sub(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>\s*", "", text, flags=re.S)
    text = re.sub(r"<\|[^|>]+\|>", "", text)
    return text.strip()


def nemotron(text: str) -> str:
    text = re.sub(r"<x_[\d.]+><y_[\d.]+>(.*?)<x_[\d.]+><y_[\d.]+><class_(Page-header|Page-footer|Picture)>",
                  "", text, flags=re.S)
    text = re.sub(r"<[xy]_[\d.]+>", "", text)
    text = re.sub(r"<class_[^>]+>", "\n\n", text)
    text = re.sub(r"</?s>|<pad>", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


FORMATS = {"markdown": markdown, "olmocr": olmocr, "dots_json": dots_json, "doctags": doctags, "html": html,
           "deepseek": deepseek, "nemotron": nemotron}


def safe(page_id: str) -> str:
    return page_id.replace("/", "__")


def main(argv):
    answers, out_dir = argv[1], argv[2]
    os.makedirs(out_dir, exist_ok=True)
    failed = 0
    with open(answers) as f:
        for line in f:
            row = json.loads(line)
            fmt = row.get("output", "markdown")
            try:
                md = FORMATS[fmt](row.get("reply") or "") if not row.get("error") else ""
            except Exception as exc:  # noqa: BLE001 - one unreadable reply is one empty page
                failed += 1
                print(f"[convert] {row['id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
                md = ""
            with open(os.path.join(out_dir, safe(row["id"]) + ".md"), "w") as out:
                out.write(md)
    print(json.dumps({"converted_failed": failed}))


if __name__ == "__main__":
    main(sys.argv)
