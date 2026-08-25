from milabench.commands import max_node_count
from milabench.utils import enumerate_rank, select_nodes


def test_enumerate_rank():
    nodes = [
        {"main": False},
        {"main": False},
        {"main": True},
        {"main": False},
    ]
    ranks = [r for r, _ in enumerate_rank(nodes)]

    assert ranks == [1, 2, 0, 3]


def test_select_nodes():
    nodes = [
        {"main": False},
        {"main": False},
        {"main": True},
        {"main": False},
    ]

    selected = select_nodes(nodes, 3)
    assert selected == [{"main": True}, {"main": False}, {"main": False}]


def _cluster(n):
    return [{"name": f"n{i}", "main": i == 0} for i in range(n)]


def test_max_node_count_defaults_to_one():
    config = {"system": {"nodes": _cluster(2)}}
    assert max_node_count(config) == 1
    assert max_node_count({**config, "num_machines": 1}) == 1


def test_max_node_count_uses_num_machines():
    config = {"system": {"nodes": _cluster(4)}, "num_machines": 2}
    assert max_node_count(config) == 2


def test_max_node_count_max_nodes_overrides_and_caps():
    nodes = _cluster(2)
    assert max_node_count({"system": {"nodes": nodes}, "num_machines": 1, "max_nodes": 2}) == 2
    assert max_node_count({"system": {"nodes": nodes}, "num_machines": 8}) == 2
