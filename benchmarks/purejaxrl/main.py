# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

import argklass
import torch  # This is a bit of a trick to make jax use torch's packaged libs

from dqn import add_dqn_command, main as dqn_main
from ppo import add_ppo_command, main as ppo_main


def main():
    parser = argklass.ArgumentParser(description="PureJaxRL")
    subparser = parser.add_subparsers(title="Benchmark", dest="benchmark")

    add_dqn_command(subparser)
    add_ppo_command(subparser)

    bench = {
        "dqn": dqn_main,
        "ppo": ppo_main
    }

    args = parser.parse_args()

    if benchmark := bench.get(args.benchmark):
        benchmark(args)

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()

    # The run is over and its metrics are out: skip interpreter/C++ teardown, where
    # XLA's runtime threads can abort with "terminate called without an active
    # exception" and turn a finished run into a failure (ppo.D0, 1x4xL40s_run_ce0bb376)
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
