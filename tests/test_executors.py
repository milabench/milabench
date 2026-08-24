import asyncio
import os
from contextlib import contextmanager

import pytest

import milabench.commands.executors
from milabench.alt_async import proceed
from milabench.commands import (
    NJobs,
    PackCommand,
    PerGPU,
    SingleCmdCommand,
    TorchRunCommand,
    VoirCommand,
)
from milabench.common import _get_multipack, CommonArguments


class ExecMock1(SingleCmdCommand):
    def __init__(self, pack_or_exec, *exec_argv, **kwargs) -> None:
        super().__init__(pack_or_exec, **kwargs)
        self.exec_argv = exec_argv

    def _argv(self, **kwargs):
        return [
            f"cmd{self.__class__.__name__}",
            *self.exec_argv,
            *[f"{self.__class__.__name__}[arg{i}]" for i in range(2)],
            *[f"{self.__class__.__name__}{k}:{v}" for k, v in kwargs.items()],
        ]


class ExecMock2(ExecMock1):
    pass


TEST_FOLDER = os.path.dirname(__file__)


def benchio():
    args = CommonArguments(
        config=os.path.join(TEST_FOLDER, "config", "benchio.yaml"),
        base="/tmp",
        use_current_env=True,
    )

    packs = _get_multipack(args=args, run_name="test", overrides={})

    _, pack = packs.packs.popitem()
    return pack


@pytest.fixture
def noexecute(monkeypatch):
    async def execute(pack, *args, **kwargs):
        return [*args, *[f"{k}:{v}" for k, v in kwargs.items()]]

    monkeypatch.setattr(milabench.commands.executors, "execute", execute)


def mock_pack(pack):
    async def execute(*args, **kwargs):
        return [*args, *[f"{k}:{v}" for k, v in kwargs.items()]]

    mock = pack
    mock.execute = execute
    return mock


def test_executor_argv():
    submock = ExecMock1(mock_pack(benchio()), "a1", "a2")
    wrapmock = ExecMock2(submock, "a3")

    assert wrapmock.argv() == [
        "cmdExecMock2",
        "a3",
        "ExecMock2[arg0]",
        "ExecMock2[arg1]",
        "cmdExecMock1",
        "a1",
        "a2",
        "ExecMock1[arg0]",
        "ExecMock1[arg1]",
    ]

    submock = ExecMock2(mock_pack(benchio()))
    wrapmock = ExecMock1(submock)

    assert wrapmock.argv(k1="v1") == [
        "cmdExecMock1",
        "ExecMock1[arg0]",
        "ExecMock1[arg1]",
        "ExecMock1k1:v1",
        "cmdExecMock2",
        "ExecMock2[arg0]",
        "ExecMock2[arg1]",
        "ExecMock2k1:v1",
    ]


def test_executor_kwargs():
    submock = ExecMock1(mock_pack(benchio()), selfk1="sv1", selfk2="sv2")
    wrapmock = ExecMock2(submock, selfk1="sv1'", selfk3="sv3")
    kwargs = {"selfk2": "v2''", "selfk3": "v3''", "k4": "v4"}

    assert sorted(wrapmock.kwargs().keys()) == ["selfk1", "selfk2", "selfk3"]
    assert sorted(wrapmock.kwargs().values()) == ["sv1'", "sv2", "sv3"]


def test_executor_execute(noexecute):
    submock = ExecMock1(mock_pack(benchio()), "a1", selfk1="sv1")
    wrapmock = ExecMock2(submock, "a2", selfk2="sv2")

    result = asyncio.run(wrapmock.execute(k3="v3"))
    expected = [
        [
            "cmdExecMock2",
            "a2",
            "ExecMock2[arg0]",
            "ExecMock2[arg1]",
            "cmdExecMock1",
            "a1",
            "ExecMock1[arg0]",
            "ExecMock1[arg1]",
            "selfk1:sv1",
            "selfk2:sv2",
            "k3:v3",
        ]
    ]
    print(result)
    print(expected)
    assert sorted(result) == sorted(expected)


def test_pack_executor():
    # voir is not setup so we are not receiving anything
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")

    acc = 0
    for r in proceed(executor.execute()):
        print(r)
        acc += 1

    assert acc >= 4, "Only 4 message received (config, meta, start, end)"


def test_voir_executor():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)

    acc = 0
    for r in proceed(voir.execute()):
        print(r)
        acc += 1

    assert acc >= 72


def test_timeout():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20", "--sleep", 20)
    voir = VoirCommand(executor)

    acc = 0
    for r in proceed(voir.execute(timeout=True, timeout_delay=1)):
        print(r)
        acc += 1

    assert acc > 2 and acc < 72


def test_njobs_executor():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    njobs = NJobs(voir, 5)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1

    assert acc >= 72 * 5


def test_njobs_gpus_executor():
    """Two GPUs so torch run IS used"""
    devices = mock_gpu_list()

    from importlib.util import find_spec

    if find_spec("torch") is None:
        pytest.skip("Pytorch is not installed")

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    torchcmd = TorchRunCommand(voir, use_stdout=True)
    njobs = NJobs(torchcmd, 1, devices)

    acc = 0
    for r in proceed(njobs.execute()):
        if r.event == "start":
            assert r.data["command"][0].endswith("benchrun")
        acc += 1
        print(r)

    assert acc >= len(devices) * 70


def test_njobs_gpu_executor():
    """One GPU, so torch run is not used"""
    devices = [mock_gpu_list()[0]]

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    torch = TorchRunCommand(voir, use_stdout=True)
    njobs = NJobs(torch, 1, devices)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)

        if r.event == "start":
            assert r.data["command"][0].endswith("voir")

        acc += 1

    assert acc >= len(devices) * 72


def test_njobs_novoir_executor():
    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    njobs = NJobs(executor, 5)

    acc = 0
    for r in proceed(njobs.execute()):
        print(r)
        acc += 1

    assert acc >= 2 * 10


def mock_gpu_list():
    return [
        {"device": 0, "selection_variable": "CUDA_VISIBLE_DEVICE"},
        {"device": 1, "selection_variable": "CUDA_VISIBLE_DEVICE"},
    ]


def test_per_gpu_executor():
    devices = mock_gpu_list()

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    voir = VoirCommand(executor)
    plan = PerGPU(voir, devices)

    acc = 0
    for r in proceed(plan.execute()):
        print(r)
        acc += 1

    assert acc >= len(devices) * 72


def test_void_executor():
    from milabench.commands import VoidCommand

    plan = VoirCommand(VoidCommand(benchio()))

    for _ in proceed(plan.execute()):
        pass


def await_now(function):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(function)


from argparse import Namespace

class MockPack:
    config = {
        "max_duration": 1,
        "name": "Mock"
    }
    processes = []

    dirs = Namespace(**{
        "code": os.getcwd()
    })

    def full_env(self, *args, **kwargs):
        return {}

    async def send(self, **kwargs):
        print(kwargs)

    async def message(self, msg):
        print(msg)

    @property
    def working_directory(self):
        return self.dirs.code

class Commands:
    def __init__(self, time) -> None:
        self.time = time

    def packs(self):
        return []

    def commands(self):
        yield MockPack(), ["sleep", str(self.time)], {}


def test_execute_command_timeout():
    from milabench.commands.executors import execute_command
    
    future = execute_command(Commands(10), timeout=True, timeout_delay=1)
    
    for msg in proceed(future):
        print(msg)



def test_execute_command():
    from milabench.commands.executors import execute_command

    future = execute_command(Commands(0), timeout=True, timeout_delay=1)
    messages = []
    for msg in proceed(future):
        messages.append(msg)

    assert len(messages) == 2
    assert messages[-1].data["return_code"] == 0


def test_count_command_errors_scalars():
    from milabench.commands import count_command_errors

    assert count_command_errors(None) == 0
    assert count_command_errors(0) == 0
    assert count_command_errors(1) == 1
    assert count_command_errors(3) == 3
    # bool is a subclass of int; make sure it stays sane
    assert count_command_errors(True) == 1
    assert count_command_errors(False) == 0


def test_count_command_errors_exceptions():
    from milabench.commands import count_command_errors

    assert count_command_errors(RuntimeError("boom")) == 1
    assert count_command_errors(ValueError()) == 1


def test_count_command_errors_list_of_none():
    """A leaf command run without timeout returns ``asyncio.gather`` results,
    which are ``None`` per successful process -> zero errors, not a TypeError."""
    from milabench.commands import count_command_errors

    # This is exactly the shape that used to trigger:
    #   TypeError: int() argument ... not 'list'
    assert count_command_errors([None, None, None]) == 0
    assert count_command_errors([]) == 0


def test_count_command_errors_nested():
    from milabench.commands import count_command_errors

    assert count_command_errors([None, RuntimeError(), 2]) == 3
    assert count_command_errors([[None], [RuntimeError()], [1, 1]]) == 3


def test_count_command_errors_futures():
    from milabench.commands import count_command_errors

    async def make_futures():
        loop = asyncio.get_running_loop()

        ok = loop.create_future()
        ok.set_result(None)

        failed = loop.create_future()
        failed.set_exception(RuntimeError("boom"))

        cancelled = loop.create_future()
        cancelled.cancel()
        # give the loop a chance to process the cancellation
        await asyncio.sleep(0)

        return ok, failed, cancelled

    ok, failed, cancelled = asyncio.run(make_futures())

    assert count_command_errors(ok) == 0
    assert count_command_errors(failed) == 1
    assert count_command_errors(cancelled) == 1
    # A list of completed tasks is what execute_command returns on timeout
    assert count_command_errors([ok, failed, cancelled]) == 2


def test_list_command_execute_aggregates_child_lists(monkeypatch):
    """Regression test: ``NJobs``/``PerGPU`` (ListCommand) parallel execute
    used to do ``int(result)`` where each child returned a *list*, raising
    ``TypeError: int() argument ... not 'list'`` for every benchmark."""
    from milabench.commands import NJobs

    async def fake_child_execute(self, *args, **kwargs):
        # Mimic a leaf command run without timeout: gather() of Nones
        return [None, None]

    monkeypatch.setattr(
        "milabench.commands.Command.execute", fake_child_execute, raising=True
    )

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    njobs = NJobs(executor, 3)

    # Pass a warden sentinel so get_or_create_warden reuses it instead of
    # spinning up a real process_cleaner (GPU warden) during the test.
    error_count = asyncio.run(njobs.execute(warden=object()))
    # Must not raise TypeError and must report zero errors for successful runs
    assert error_count == 0


def test_list_command_execute_counts_child_errors(monkeypatch):
    from milabench.commands import NJobs

    async def fake_child_execute(self, *args, **kwargs):
        # One failed process (exception) among the gathered results
        return [None, RuntimeError("child failed")]

    monkeypatch.setattr(
        "milabench.commands.Command.execute", fake_child_execute, raising=True
    )

    executor = PackCommand(benchio(), "--start", "2", "--end", "20")
    njobs = NJobs(executor, 2)

    error_count = asyncio.run(njobs.execute(warden=object()))
    # 2 jobs, each reporting 1 error in its result list
    assert error_count == 2


def test_get_or_create_warden_reuses_existing_warden():
    """A child execution must not open its own process_cleaner: it kills
    every process using the GPU on enter/exit, so opening it more than once
    for one execution tree makes concurrent siblings kill each other."""
    from milabench.commands.executors import get_or_create_warden

    sentinel = object()
    with get_or_create_warden(sentinel) as warden:
        assert warden is sentinel


def test_get_or_create_warden_creates_one_when_root(monkeypatch):
    import milabench.commands.executors as executors_mod

    created = []

    @contextmanager
    def fake_process_cleaner(**kwargs):
        obj = object()
        created.append(obj)
        yield obj

    monkeypatch.setattr(executors_mod, "process_cleaner", fake_process_cleaner)

    with executors_mod.get_or_create_warden(None) as warden:
        assert warden is created[0]

    assert len(created) == 1


def test_sequence_command_shares_one_warden_across_children(monkeypatch):
    """SequenceCommand.execute() recurses into each child's own execute(),
    which used to each open a fresh process_cleaner (killing every other
    GPU process on enter/exit -- see get_or_create_warden's docstring).
    A single warden must now be shared across the whole tree."""
    from milabench.commands import SequenceCommand
    import milabench.commands.executors as executors_mod

    calls = []
    real_process_cleaner = executors_mod.process_cleaner

    @contextmanager
    def counting_process_cleaner(*args, **kwargs):
        calls.append(1)
        with real_process_cleaner(*args, **kwargs) as warden:
            yield warden

    monkeypatch.setattr(executors_mod, "process_cleaner", counting_process_cleaner)

    executor1 = PackCommand(benchio(), "--start", "2", "--end", "5")
    executor2 = PackCommand(benchio(), "--start", "2", "--end", "5")
    plan = SequenceCommand(VoirCommand(executor1), VoirCommand(executor2))

    for _ in proceed(plan.execute()):
        pass

    assert len(calls) == 1
