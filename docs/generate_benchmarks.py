#!/usr/bin/env python3
"""Generate per-benchmark RST cheatsheet pages from config YAML and a Jinja2 template.

Run from the docs/ directory before building Sphinx:

    python generate_benchmarks.py

Generated files land in docs/Benchmarks/<name>.rst and are .gitignored.
Hand-written content lives in:
  - benchmarks/<group>/README.md        (group overview, shared across benchmarks in the group)
  - benchmarks/<group>/<bench_name>.md  (cheatsheet for a specific benchmark)
"""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader


DOCS_DIR = Path(__file__).resolve().parent
REPO_DIR = DOCS_DIR.parent
CONFIG_DIR = REPO_DIR / "config"
BENCHMARKS_DIR = REPO_DIR / "benchmarks"
OUTPUT_DIR = DOCS_DIR / "Benchmarks"

# Suffixes that distinguish scale/precision variants of the same logical benchmark.
_VARIANT_SUFFIXES = re.compile(
    r"-(fp8|fp16|fp32|bf16|tf32|tf32-fp16|noio"
    r"|single|gpus|nodes"
    r"|ddp-gpus|ddp-nodes|mp-gpus|mp-nodes"
    r")$"
)


# ---------------------------------------------------------------------------
# Lightweight config loading (no milabench dependency)
# ---------------------------------------------------------------------------

def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml_with_includes(filepath: Path) -> dict:
    """Load a YAML file, recursively resolving ``include:`` directives."""
    with open(filepath) as f:
        data = yaml.safe_load(f) or {}

    includes = data.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]

    merged: dict = {}
    for inc in includes:
        inc_path = (filepath.parent / inc).resolve()
        merged = deep_merge(merged, load_yaml_with_includes(inc_path))

    merged = deep_merge(merged, data)
    return merged


def resolve_inheritance(configs: dict) -> dict:
    """Resolve ``inherits:`` chains in-place and return only public benchmarks."""
    resolved: dict = {}

    def _resolve(name: str) -> dict:
        if name in resolved:
            return resolved[name]

        cfg = copy.deepcopy(configs[name])
        parent_name = cfg.pop("inherits", None)
        if parent_name and parent_name in configs:
            parent = _resolve(parent_name)
            tags = sorted({*parent.get("tags", []), *cfg.get("tags", [])})
            cfg = deep_merge(parent, cfg)
            cfg["tags"] = tags

        resolved[name] = cfg
        return cfg

    for name in configs:
        _resolve(name)

    return {
        name: cfg
        for name, cfg in resolved.items()
        if not name.startswith("_") and name != "*"
    }


def load_all_benchmarks(config_path: Path) -> dict:
    raw = load_yaml_with_includes(config_path)
    return resolve_inheritance(raw)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def definition_to_group(definition: str) -> str:
    """Extract the benchmark folder name from a definition path."""
    return Path(definition).name


def bench_base_name(name: str) -> str:
    """Strip variant suffixes to get the logical benchmark base name.

    e.g. ``convnext_large-fp16`` -> ``convnext_large``
         ``llm-lora-ddp-gpus``   -> ``llm-lora``
         ``resnet50``            -> ``resnet50``
    """
    base = name
    while True:
        stripped = _VARIANT_SUFFIXES.sub("", base)
        if stripped == base:
            break
        base = stripped
    return base


def group_by_base(benchmarks: dict) -> dict[str, list[dict]]:
    """Group benchmarks by (group_folder, base_name).

    Returns ``{base_name: [bench_dict, ...]}`` with a ``"name"`` key injected.
    """
    bases: dict[str, list[dict]] = {}
    for name, cfg in benchmarks.items():
        definition = cfg.get("definition")
        if not definition:
            continue
        base = bench_base_name(name)
        cfg_with_name = {**cfg, "name": name}
        bases.setdefault(base, []).append(cfg_with_name)

    return dict(sorted(bases.items()))


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------

def pick_primary(variants: list[dict]) -> dict:
    weighted = [v for v in variants if v.get("weight", 0) > 0]
    if weighted:
        return weighted[0]
    enabled = [v for v in variants if v.get("enabled", False)]
    if enabled:
        return enabled[0]
    return variants[0]


def find_cheatsheet(group: str, base_name: str, variants: list[dict]) -> Path | None:
    """Look for a cheatsheet .md for this benchmark.

    Search order:
      1. benchmarks/<group>/<base_name>.md   (e.g. benchmarks/torchvision/resnet50.md)
      2. benchmarks/<group>/<variant_name>.md (first match)
    """
    candidate = BENCHMARKS_DIR / group / f"{base_name}.md"
    if candidate.is_file():
        return candidate

    for v in variants:
        candidate = BENCHMARKS_DIR / group / f"{v['name']}.md"
        if candidate.is_file():
            return candidate

    return None


def render_benchmark(base_name: str, variants: list[dict], env: Environment) -> str:
    template = env.get_template(".template.rst.j2")

    primary = pick_primary(variants)
    definition = primary.get("definition", "")
    group = definition_to_group(definition)
    bench_group = primary.get("group", group)

    title = base_name

    readme_path_abs = BENCHMARKS_DIR / group / "README.md"
    readme_rel = os.path.relpath(readme_path_abs, OUTPUT_DIR)
    readme_exists = readme_path_abs.is_file()

    cheatsheet_path_abs = find_cheatsheet(group, base_name, variants)
    cheatsheet_rel = os.path.relpath(cheatsheet_path_abs, OUTPUT_DIR) if cheatsheet_path_abs else ""
    cheatsheet_exists = cheatsheet_path_abs is not None

    argv = primary.get("argv", {})
    if isinstance(argv, list):
        argv = {str(a): "" for a in argv}

    plan = primary.get("plan", {})
    plan_method = plan.get("method", "") if isinstance(plan, dict) else ""

    url = primary.get("url", "")
    if isinstance(url, list):
        url = ", ".join(url)

    variant_rows = []
    for v in sorted(variants, key=lambda v: v["name"]):
        vplan = v.get("plan", {})
        variant_rows.append({
            "name": v["name"],
            "plan": vplan.get("method", "") if isinstance(vplan, dict) else "",
            "tags": v.get("tags", []),
            "num_machines": v.get("num_machines", 1),
            "enabled": v.get("enabled", False),
            "weight": v.get("weight", 0),
        })

    select_name = primary["name"]

    return template.render(
        title=title,
        definition=definition,
        group=bench_group,
        install_group=primary.get("install_group", ""),
        tags=primary.get("tags", []),
        max_duration=primary.get("max_duration", 600),
        plan_method=plan_method,
        num_machines=primary.get("num_machines", 1),
        url=url,
        argv=argv,
        variants=variant_rows,
        select_name=select_name,
        readme_path=readme_rel,
        readme_exists=readme_exists,
        cheatsheet_path=cheatsheet_rel,
        cheatsheet_exists=cheatsheet_exists,
    )


_GROUP_DISPLAY_NAMES = {
    "torchvision": "Torchvision",
    "torchvision_ddp": "Torchvision DDP",
    "huggingface": "HuggingFace",
    "timm": "timm",
    "lightning": "Lightning",
    "diffusion": "Diffusion",
    "dinov2": "DINOv2",
    "llm": "LLM (torchtune)",
    "llama": "Llama (inference)",
    "llava": "LLaVA",
    "vjepa": "V-JEPA",
    "brax": "Brax",
    "purejaxrl": "PureJaxRL",
    "cleanrl_jax": "CleanRL JAX",
    "geo_gnn": "Geometric GNN",
    "recursiongfn": "RecursionGFN",
    "rlhf": "RLHF",
    "inference": "Inference (HF)",
    "vllm": "vLLM",
    "flops": "Synthetic FLOPS",
}


def _generate_group_pages(
    group_to_sections: dict[str, list[str]],
    env: Environment,
):
    """Generate one page per group with benchmark sections inlined."""
    group_template = env.get_template(".group.rst.j2")
    group_pages = []

    for group in sorted(group_to_sections):
        display = _GROUP_DISPLAY_NAMES.get(group, group)
        page_name = f"group_{group}"
        group_pages.append(page_name)

        readme_path_abs = BENCHMARKS_DIR / group / "README.md"
        readme_rel = os.path.relpath(readme_path_abs, OUTPUT_DIR)
        readme_exists = readme_path_abs.is_file()

        rst = group_template.render(
            title=display,
            readme_path=readme_rel,
            readme_exists=readme_exists,
            select_name=group,
            benchmark_sections=group_to_sections[group],
        )

        (OUTPUT_DIR / f"{page_name}.rst").write_text(rst)

    master_lines = [
        ".. toctree::",
        "   :caption: Benchmarks",
        "   :maxdepth: 2",
        "",
    ]
    for page in group_pages:
        master_lines.append(f"   Benchmarks/{page}")
    master_lines.append("")

    (DOCS_DIR / "_benchmarks_toc.rst").write_text("\n".join(master_lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(CONFIG_DIR / "all.yaml"),
        help="Top-level config YAML (default: config/all.yaml)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for f in OUTPUT_DIR.glob("*.rst"):
        if f.name.startswith("."):
            continue
        f.unlink()

    benchmarks = load_all_benchmarks(Path(args.config))
    bases = group_by_base(benchmarks)

    env = Environment(
        loader=FileSystemLoader(str(OUTPUT_DIR)),
        keep_trailing_newline=True,
    )

    missing_cheatsheets = []
    group_to_sections: dict[str, list[str]] = {}

    for base_name, variants in bases.items():
        rst_section = render_benchmark(base_name, variants, env)

        group = definition_to_group(variants[0].get("definition", ""))
        group_to_sections.setdefault(group, []).append(rst_section)
        variant_names = ", ".join(v["name"] for v in variants)
        print(f"  {base_name}  [{group}] ({variant_names})")

        cs = find_cheatsheet(group, base_name, variants)
        if cs is None:
            missing_cheatsheets.append(f"benchmarks/{group}/{base_name}.md")

    _generate_group_pages(group_to_sections, env)

    print(f"\nGenerated {len(group_to_sections)} group pages in {OUTPUT_DIR.relative_to(DOCS_DIR)}/")

    if missing_cheatsheets:
        print(f"\nCheatsheet .md files not yet written ({len(missing_cheatsheets)}):")
        for p in missing_cheatsheets:
            print(f"  {p}")


def generate_gpu_table():
    """Fetch GPU summary from the milabench dashboard and write an RST include file."""
    import json
    from urllib.request import urlopen
    from urllib.error import URLError

    url = "https://www.milabench.com/api/gpu/summary"
    out = DOCS_DIR / "_gpu_summary.rst"

    print(f"Fetching GPU summary from {url} ...")

    try:
        with urlopen(url, timeout=10) as resp:
            rows = json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as exc:
        print(f"  Warning: could not fetch GPU summary: {exc}")
        out.write_text(
            ".. note::\n\n"
            "   GPU summary table could not be generated (dashboard unreachable).\n"
            "   See `milabench.com <https://www.milabench.com>`_ for the latest results.\n"
        )
        return

    if not rows:
        out.write_text("*No GPU data available.*\n")
        return

    def strip_quotes(s):
        if not s:
            return "-"
        return str(s).strip('"')

    def fmt_date(s):
        if not s:
            return "-"
        return s[:10]

    lines = [
        ".. list-table:: Latest GPU Runs",
        "   :header-rows: 1",
        "   :widths: 18 8 8 8 10 10 10 8 10 10",
        "",
        "   * - GPU",
        "     - GPUs",
        "     - CPU",
        "     - Arch",
        "     - PyTorch",
        "     - CUDA / ROCm",
        "     - Milabench",
        "     - Contributor",
        "     - Passed",
        "     - Last Tested",
    ]

    for row in rows:
        gpu = strip_quotes(row.get("gpu"))
        gpu_count = row.get("gpu_count", "-")
        gpu_mem = row.get("gpu_memory")
        mem_str = f" ({round(gpu_mem / 1024)} GB)" if gpu_mem else ""
        cpu_arch = row.get("cpu_arch", "-")
        arch = strip_quotes(row.get("arch", "")).upper()
        pytorch = strip_quotes(row.get("pytorch"))
        accel = strip_quotes(row.get("accel_version"))
        tag = strip_quotes(row.get("milabench_tag"))
        contributor = strip_quotes(row.get("contributor"))
        passed = row.get("passed", 0)
        total = row.get("total", 0)
        pct = f"{passed}/{total}" if total else "-"
        date = fmt_date(row.get("latest_date"))

        lines.append(f"   * - {gpu}")
        lines.append(f"     - {gpu_count}x{mem_str}")
        lines.append(f"     - {cpu_arch}")
        lines.append(f"     - {arch}")
        lines.append(f"     - {pytorch}")
        lines.append(f"     - {accel}")
        lines.append(f"     - {tag}")
        lines.append(f"     - {contributor}")
        lines.append(f"     - {pct}")
        lines.append(f"     - {date}")

    lines.append("")
    out.write_text("\n".join(lines))
    print(f"  Wrote {len(rows)} rows to {out.relative_to(DOCS_DIR)}")


if __name__ == "__main__":
    main()
    generate_gpu_table()
