# sdiffusion — diffusion fine-tuning tool comparison

Three packs wrapping the major Stable Diffusion fine-tuning frameworks so
they can be compared apples-to-apples on milabench hardware:

| pack         | tool                                                        | entry point                              |
|--------------|-------------------------------------------------------------|------------------------------------------|
| `aitoolkit`  | [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)   | `python run.py <job.yaml>`               |
| `simpletuner`| [bghira/SimpleTuner](https://github.com/bghira/SimpleTuner) | `python train_sdxl.py ...`               |
| `kohya`      | [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (a.k.a. "Kohya-ss") | `accelerate launch sdxl_train_network.py ...` |

Note: "Kohya-ss" and "sd-scripts" are the same project — the GitHub org is
`kohya-ss` and the repo is `sd-scripts`, so it is wrapped once, as `kohya`.

## Common ground

Shared helpers (repo vendoring, dataset materialization, timed runs, metric
pushing) live in `benchmate.diffusion`.

All three run the same job by default, so throughput numbers are comparable:

* **Task**: SDXL LoRA fine-tuning (`stabilityai/stable-diffusion-xl-base-1.0`,
  rank 16, constant LR 1e-4, bf16, 1024px, fixed 200 steps, bs=1)
* **Dataset**: `lambdalabs/naruto-blip-captions` — the same dataset the existing
  `diffusion` benchmark already uses (so it is cached on the clusters).
  `prepare.py` materializes a 1000-img train / 32-img val subset as
  `<image>.jpg` + `<image>.txt` caption files, which all three tools consume
  natively (kohya: standard imagefolder, ai-toolkit: `caption_ext: txt`,
  SimpleTuner: `--caption_strategy textfile`).
* **Isolation**: each pack has its own `install_group` venv
  (`sdiffusion-<tool>`) because the tools pin conflicting
  torch/diffusers/transformers versions. The trainer repos are cloned into
  `{milabench_cache}/sdiffusion/vendor/` and their own `requirements.txt`
  installed at prepare time.

## Usage

    milabench install config/sdiffusion.yaml
    milabench prepare config/sdiffusion.yaml
    milabench run config/sdiffusion.yaml

Each pack has a `dev.yaml` + `Makefile` for local iteration, e.g.
`make -C kohya single`.

Pin a tool revision for reproducibility:

    milabench prepare config/sdiffusion.yaml --kohya.prepare -- --rev v0.8.7

(see each pack's `prepare.py` for the full option list)

## Metrics

`main.py` measures wall-clock for the fixed step count and pushes
`steps_per_second` / `samples_per_second` through the benchmate stdout
protocol, tagging each run with `tool=<name>`.

## Known limitations / TODO

* Single GPU only for now (`plan: per_gpu`); multigpu/multinode would use
  each tool's native `accelerate`/DDP launch and needs per-tool wiring.
* ai-toolkit and SimpleTuner config/CLI surfaces drift between releases —
  pin `--rev` in the prepare config and adjust `build_job` /
  `build_command` if a new revision renames options.
* Only final-throughput is reported; a per-step stream (earlystop-friendly)
  would require parsing each trainer's tqdm output.
