"""Generate the shell commands milabench would run for each pack, without
actually executing anything -- useful for inspecting/debugging exactly what
a benchmark's install/run plan resolves to, standalone (no async
orchestration, no real GPUs required).

Restored and ported to argklass (see argklass.command.Command) from the
pre-#24 "Remove Coleo" version; git history has the original coleo-based
implementation for reference (milabench/cli/dry.py @ 86e58bf^). It predates
the tools/ subcommand grouping, which is why it wasn't already here.
"""
import os
import shlex
from contextlib import contextmanager
from dataclasses import dataclass

import voir.instruments.gpu as voirgpu
import yaml
from argklass.command import Command

from ...commands import disable_voir
from ...common import CommonArguments, get_multipack
from ...multi import make_execution_plan
from ...network import enable_offline
from ...sizer import resolve_argv


class MockDeviceSMI:
    def __init__(self, ngpu, capacity) -> None:
        self.devices = [i for i in range(ngpu)]
        self.used = [1]
        self.total = [capacity]
        self.util = [40]
        self.temp = [35]
        self.power = [225]

    def get_gpu_info(self, device):
        return {
            "device": device,
            "product": "MockDevice",
            "memory": {
                "used": self.used[0] // (1024**2),
                "total": self.total[0] // (1024**2),
            },
            "utilization": {
                "compute": float(self.util[0]) / 100,
                "memory": self.used[0] / self.total[0],
            },
            "temperature": self.temp[0],
            "power": self.power[0],
            "selection_variable": "CUDA_VISIBLE_DEVICES",
        }

    @property
    def arch(self):
        return "mock"

    @property
    def visible_devices(self):
        return os.environ.get("CUDA_VISIBLE_DEVICES", None)

    def get_gpus_info(self, selection=None):
        gpus = dict()
        for device in self.devices:
            if (selection is None) or (selection and str(device) in selection):
                gpus[device] = self.get_gpu_info(device)

        return gpus

    def close(self):
        pass


@contextmanager
def assume_gpu(ngpu=1, capacity=80000, enabled=False):
    if enabled:
        old = voirgpu.DEVICESMI
        voirgpu.DEVICESMI = MockDeviceSMI(ngpu, capacity)
        yield
        voirgpu.DEVICESMI = old
    else:
        yield


class BashGenerator:
    def __init__(self) -> None:
        self.indent = 0
        self.background_mode = False
        self.print("#!/bin/sh")
        self.print("")

    def print(self, *args, **kwargs):
        print("  " * self.indent, end="")
        print(*args, **kwargs)

    def section(self, title):
        self.echo("---")
        self.echo(title)
        self.echo("=" * len(title))

    def echo(self, msg):
        self.print(f'echo "{msg}"')

    def comment(self, cmt):
        self.print(f"# {cmt}")

    def env(self, env):
        for k, v in env.items():
            self.print(f"export {k}={shlex.quote(v)}")
        self.print()

    @contextmanager
    def subshell(self):
        self.print("time (")
        self.indent += 1
        yield
        self.indent -= 1
        self.print(")")

    @contextmanager
    def background(self):
        self.background_mode = True
        yield
        self.print("wait")
        self.background_mode = False

    def command(self, *args, env=None, **kwargs):
        prefix = []
        if env is not None:
            for k, v in env.items():
                prefix.append(f"{k}={v}")

        prefix = " ".join(prefix)
        sufix = "&"

        frags = [prefix] + [str(a) for a in args] + [sufix]

        self.print(" ".join(filter(lambda x: x != "", frags)))


# fmt: off
@dataclass
class Arguments:
    """Generate dry commands to execute the bench standalone."""
    nnodes  : int  = 2      # Number of nodes to simulate
    ngpu    : int  = 8      # Number of GPUs to simulate per node
    capacity: int  = 80000  # Simulated GPU memory capacity (MiB)
    withenv : bool = True   # Print venv activation + milabench env setup
    usevoir : bool = False  # Keep voir enabled instead of disabling it
    ncpu    : int  = 256    # Simulated CPU core count
# fmt: on


def _multipack_args(conf: Arguments) -> CommonArguments:
    args = CommonArguments()
    args.system = "system_tmp.yaml"

    system = {
        "system": {
            "arch": "cuda",
            "nodes": [
                {
                    "name": str(i),
                    "ip": f"192.168.0.{i + 10}" if i != 0 else "127.0.0.1",
                    "user": "username",
                    "main": i == 0,
                    "sshport": 22,
                }
                for i in range(conf.nnodes)
            ],
        }
    }

    with open("system_tmp.yaml", "w") as file:
        file.write(yaml.dump(system))

    return args


@contextmanager
def with_env(**kwargs):
    for k, v in kwargs.items():
        if v:
            os.environ[k] = str(v)

    yield

    for k, v in kwargs.items():
        if v:
            del os.environ[k]


def cli_dry(args=None):
    """Generate dry commands to execute the bench standalone"""
    if args is None:
        args = Arguments()

    with disable_voir(enabled=False), enable_offline(enabled=True):
        with assume_gpu(args.ngpu, args.capacity, enabled=True), with_env(MILABENCH_CPU_TOTAL_COUNT=args.ncpu):
            repeat = 1
            mp = get_multipack(_multipack_args(args), run_name="dev")
            gen = BashGenerator()

            first_pack = True
            for index in range(repeat):
                for pack in mp.packs.values():
                    if first_pack and args.withenv:
                        first_pack = False
                        gen.section("Virtual Env")

                        venv = pack.core._nox_session.env["VIRTUAL_ENV"]
                        gen.env({"VIRTUAL_ENV": venv})
                        gen.print("source $VIRTUAL_ENV/bin/activate")

                        gen.section("Milabench")
                        gen.env(pack.make_env())

                    exec_plan = make_execution_plan(pack, index, repeat)

                    gen.section(pack.config["name"])
                    with gen.subshell():
                        with gen.background():
                            for pack, argv, _ in exec_plan.commands():
                                final_args = resolve_argv(pack, argv)

                                gen.command(*final_args, env=pack.config.get("env", {}))

                    print()

        try:
            os.remove("system_tmp.yaml")
        except OSError:
            pass


class Dry(Command):
    """Generate dry commands to execute the bench standalone."""

    name = "dry"
    Arguments = Arguments

    @staticmethod
    def execute(args):
        cli_dry(args)


COMMANDS = Dry
