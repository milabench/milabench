import time

import pytest

from benchmate.nodemon.sink import TsvSink


def test_flushes_are_shifted_by_up_to_the_jitter(tmp_path):
    sink = TsvSink(tmp_path, "cn-a", flush_every_s=10.0)  # default 2.5%: ±250 ms
    delays = []
    for _ in range(50):
        before = time.monotonic()
        sink.flush()
        delays.append(sink.next_flush_monotonic_s - before)

    assert all(9.75 <= delay <= 10.25 + 0.1 for delay in delays)
    # Drawn afresh each time: two samplers started together drift apart.
    assert len({round(delay, 6) for delay in delays}) > 1
    sink.close()


def test_the_jitter_scales_with_the_period_so_no_flush_is_due_in_the_past(tmp_path):
    sink = TsvSink(tmp_path, "cn-a", flush_every_s=0.1, flush_jitter_frac=0.25)
    for _ in range(200):
        before = time.monotonic()
        sink.flush()
        assert 0.075 <= sink.next_flush_monotonic_s - before <= 0.125 + 0.1
    sink.close()


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_a_jitter_of_a_whole_period_or_negative_is_refused(tmp_path, fraction):
    with pytest.raises(ValueError, match="flush_jitter_frac"):
        TsvSink(tmp_path, "cn-a", flush_jitter_frac=fraction)


def test_no_jitter_flushes_exactly_every_period(tmp_path):
    sink = TsvSink(tmp_path, "cn-a", flush_every_s=10.0, flush_jitter_frac=0.0)
    before = time.monotonic()
    sink.flush()

    assert 10.0 <= sink.next_flush_monotonic_s - before <= 10.1
    sink.close()
