import argparse
import logging
from contextlib import nullcontext

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchcompat.core as accelerator

import transformers
from benchmate.observer import BenchObserver

from .models import models
from .synth import SyntheticData, generators

log = logging.getLogger(__name__)

# -v / -vv / -vvv / ... → increasingly verbose (index into this tuple).
_VERBOSITY_LEVELS = (
    logging.CRITICAL,
    logging.ERROR,
    logging.WARNING,
    logging.INFO,
    logging.DEBUG,
)
_MAX_VERBOSITY = len(_VERBOSITY_LEVELS) - 1


def configure_logging(verbosity: int) -> int:
    """Configure root logging from a ``-v`` count (more ``-v`` → more verbose)."""
    level = _VERBOSITY_LEVELS[min(max(verbosity, 0), _MAX_VERBOSITY)]

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    log.debug(
        "logging configured: verbosity=%s level=%s",
        verbosity,
        logging.getLevelName(level),
    )
    return level


def is_tf32_allowed(args):
    return "tf32" in args.precision


def is_fp16_allowed(args):
    return "fp16" in args.precision or "bf16" in args.precision


def float_dtype(precision):
    if "fp16" in precision:
        # CUDA commonly uses fp16; other backends prefer bf16 for stability.
        if accelerator.device_type == "cuda":
            return torch.float16
        return torch.bfloat16

    if "bf16" in precision:
        return torch.bfloat16

    return torch.float


def uses_grad_scaler(args) -> bool:
    """GradScaler is mainly useful for fp16 on CUDA/XPU; skip elsewhere."""
    if not is_fp16_allowed(args):
        return False
    if "bf16" in args.precision:
        return False
    return accelerator.device_type in ("cuda", "xpu")


class NoScale:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        accelerator.optimizer_step(optimizer)

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass


def make_dataloader(args, info):
    log.info(
        "building synthetic dataloader: batch_size=%s num_workers=%s category=%s",
        args.batch_size,
        args.num_workers,
        info.category,
    )
    data = SyntheticData(
        n=args.batch_size,
        repeat=100000,
        generators=generators[info.category](info),
    )
    return DataLoader(
        data, batch_size=args.batch_size, num_workers=args.num_workers
    )


class Runner:
    def __init__(self, args):
        log.info(
            "initializing runner: model=%s precision=%s batch_size=%s lr=%s seed=%s",
            args.model,
            args.precision,
            args.batch_size,
            args.lr,
            args.seed,
        )
        log.info(
            "accelerator: device_type=%s tf32_allowed=%s",
            accelerator.device_type,
            is_tf32_allowed(args),
        )

        accelerator.set_enable_tf32(is_tf32_allowed(args))

        log.debug("setting seeds: %s", args.seed)
        accelerator.manual_seed(args.seed)
        transformers.set_seed(args.seed)
        dtype = float_dtype(args.precision)

        self.device = accelerator.fetch_device(0)
        log.info("using device: %s", self.device)

        self.batch_size = args.batch_size
        log.info("loading model info for %s", args.model)
        self.info = models[args.model](args)
        log.info(
            "model info ready: category=%s model_type=%s",
            getattr(self.info, "category", None),
            type(self.info.model).__name__,
        )

        log.info("moving model to device")
        self.model = self.info.model.to(self.device, dtype=dtype)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        log.debug("optimizer: Adam lr=%s", args.lr)

        # this cause the bench to fail for one model (reformer)
        # dtype=float_dtype(args.precision)
        log.info("applying accelerator.optimize")
        self.model, self.optimizer = accelerator.optimize(
            self.model, optimizer=self.optimizer
        )

        if hasattr(self.info, "dataloader"):
            log.info("using model-provided dataloader")
            self.loader = self.info.dataloader
        else:
            self.loader = make_dataloader(args, self.info)

        self.amp_scaler = NoScale()
        if uses_grad_scaler(args):
            log.info("enabling GradScaler for %s", accelerator.device_type)
            self.amp_scaler = accelerator.amp.GradScaler(enabled=True)
        else:
            log.debug("GradScaler disabled")

        if is_fp16_allowed(args):
            log.info("AMP autocast enabled: dtype=%s", dtype)
            self.amp_context = lambda: accelerator.amp.autocast(dtype=dtype)
        else:
            log.debug("AMP autocast disabled")
            self.amp_context = nullcontext

        log.info("runner initialization complete")

    def step(self, data):
        self.optimizer.zero_grad()

        with self.amp_context():
            outputs = self.model(**data)

        loss = outputs.loss
        log.debug("step loss=%s", loss)
        self.amp_scaler.scale(loss).backward()
        self.amp_scaler.unscale_(self.optimizer)
        # Routes through accelerator so TT/XLA flush via mark_step when needed.
        accelerator.optimizer_step(self.optimizer)
        self.amp_scaler.update()
        return loss

    def train(self):
        log.info("starting training loop")

        def batch_size(bs):
            # whisper: ['input_features', 'labels']
            # bert   : ['input_ids', 'labels']
            input_ids = bs.get("labels")
            if input_ids is not None:
                return input_ids.shape[0]

            log.error("batch size unknown; keys=%s", list(bs.keys()))
            raise RuntimeError("Batch size unknown")

        observer = BenchObserver(
            event_fn=accelerator.Event,
            batch_size_fn=batch_size,
        )
        loader = observer.loader(self.loader)
        log.debug("BenchObserver loader ready")

        step = 0
        for data in loader:
            log.debug("preparing batch step=%s keys=%s", step, list(data.keys()))
            data = accelerator.prepare_batch(**data)
            loss = self.step(data)
            observer.record_loss(loss)
            step += 1

        log.info("training loop finished after %s steps", step)


def parser():
    parser = argparse.ArgumentParser(description="Transformers models")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--model", type=str, help="Transformer model name", required=True
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training (legacy; prefer torchcompat device selection)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        metavar="S",
        help="random seed (default: 1234)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32", "tf32", "tf32-fp16", "bf16"],
        default="fp32",
        help="Precision configuration",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        # Highest verbosity for now; later lower this toward 0.
        default=_MAX_VERBOSITY,
        help=(
            "increase logging verbosity (repeatable: -v, -vv, …); "
            "maps 0..4 → CRITICAL..DEBUG; defaults to 4 (DEBUG)"
        ),
    )
    return parser


def main():
    args = parser().parse_args()
    configure_logging(args.verbose)
    log.info("parsed args: %s", vars(args))
    runner = Runner(args)
    runner.train()


if __name__ == "__main__":
    main()
