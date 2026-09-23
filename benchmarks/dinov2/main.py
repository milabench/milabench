#!/usr/bin/env python

import os

# xFormers has no XPU kernels; make dinov2 fall back to its native PyTorch
# attention (MemEffAttention -> Attention.forward). Scoped to this pack only.
os.environ.setdefault("XFORMERS_DISABLED", "1")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(__file__) + "/src/")

    from dinov2.train.train import main, get_args_parser
    args = get_args_parser(add_help=True).parse_args()
    main(args)
