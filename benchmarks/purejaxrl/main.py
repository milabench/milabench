# This is the script run by milabench run (by default)

# It is possible to use a script from a GitHub repo if it is cloned using
# clone_subtree in the benchfile.py, in which case this file can simply
# be deleted.

import argklass

import torch  # This is a bit of a trick to make jax use torch's packaged libs

from dqn import add_dqn_command, main as dqn_main

try:
    from ppo import add_ppo_command, main as ppo_main
    PPO_IMPORT_ERROR = None
except Exception as exc:  # Keep DQN usable when PPO deps are incompatible.
    add_ppo_command = None
    ppo_main = None
    PPO_IMPORT_ERROR = exc


def main():
    parser = argklass.ArgumentParser(description="PureJaxRL")
    subparser = parser.add_subparsers(title="Benchmark", dest="benchmark")

    add_dqn_command(subparser)
    if add_ppo_command is not None:
        add_ppo_command(subparser)

    bench = {
        "dqn": dqn_main,
    }
    if ppo_main is not None:
        bench["ppo"] = ppo_main

    args = parser.parse_args()

    if args.benchmark == "ppo" and PPO_IMPORT_ERROR is not None:
        raise RuntimeError("PPO is unavailable in this environment") from PPO_IMPORT_ERROR

    if benchmark := bench.get(args.benchmark):
        benchmark(args)

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
