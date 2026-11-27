"""omni_parse_bench unit tests on Markdown pages; runs in the docparse-score venv (omni-parse-bench installed).

    python score_opb.py <items.jsonl> <md_dir> <data_root> <tag> <out.json>

Each page's Markdown goes through omni-parse-bench's own converter (``markdown.to_html``, the one every provider's
Markdown goes through) and its unit-test scorer. Only ``html`` tests are scored: ``layout_kind`` tests need layout
blocks, which a Markdown page has not (the benchmark's per-tag "hard slice" leaves them out the same way). The
score is the pass rate of the tests carrying ``tag`` (``fra``, ``handwriting``); ``all`` is the pass rate of every
html test of the slice's pages.
"""

from __future__ import annotations

import json
import os
import sys

from omni_parse_bench import markdown
from omni_parse_bench.scorers import unit_tests


def main(argv):
    items_path, md_dir, data_root, tag, out = argv[1:6]
    with open(items_path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    tagged = [0, 0]
    every = [0, 0]
    pages = []
    for item in items:
        md_path = os.path.join(md_dir, item["id"].replace("/", "__") + ".md")
        md = open(md_path).read() if os.path.exists(md_path) else None
        tests = [t for t in unit_tests.tests.read(os.path.join(data_root, item["meta"]["gold"]))
                 if t.output_type == "html"]
        pred = {"html": markdown.to_html(md) if md is not None else None}
        verdicts = {v.id: v.result == "pass" for v in unit_tests.score(pred, tests)}
        page = {"id": item["id"], "tests": len(tests), "passed": sum(verdicts.values()), "tagged": 0, "tagged_passed": 0}
        for t in tests:
            ok = verdicts.get(t.id, False)
            every[0] += ok
            every[1] += 1
            if tag in t.tags:
                tagged[0] += ok
                tagged[1] += 1
                page["tagged"] += 1
                page["tagged_passed"] += ok
        pages.append(page)
    result = {"tag": tag, "tagged_passed": tagged[0], "tagged_tests": tagged[1],
              "pass_rate": tagged[0] / tagged[1] if tagged[1] else None,
              "all_passed": every[0], "all_tests": every[1],
              "all_pass_rate": every[0] / every[1] if every[1] else None, "pages": pages}
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}))


if __name__ == "__main__":
    main(sys.argv)
