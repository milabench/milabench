from benchmate.metrics import agree_observation_count


def test_budget_to_count():
    assert agree_observation_count(300, 4.6) == 66  # ceil(300 / 4.6)


def test_at_least_one_observation():
    assert agree_observation_count(1, 10.0) == 1
    assert agree_observation_count(300, 0) == 1


def test_max_count_caps():
    assert agree_observation_count(300, 0.1, max_count=100) == 100
