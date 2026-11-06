import time

from benchmate.nodemon.sink import TsvSink


def test_flushes_are_shifted_by_up_to_the_jitter(tmp_path):
    sink = TsvSink(tmp_path, "cn-a", flush_every_s=10.0, flush_jitter_s=0.25)
    delays = []
    for _ in range(50):
        before = time.monotonic()
        sink.flush()
        delays.append(sink.next_flush_monotonic_s - before)

    assert all(9.75 <= delay <= 10.25 + 0.1 for delay in delays)
    # Drawn afresh each time: two samplers started together drift apart.
    assert len({round(delay, 6) for delay in delays}) > 1
    sink.close()


def test_no_jitter_flushes_exactly_every_period(tmp_path):
    sink = TsvSink(tmp_path, "cn-a", flush_every_s=10.0, flush_jitter_s=0.0)
    before = time.monotonic()
    sink.flush()

    assert 10.0 <= sink.next_flush_monotonic_s - before <= 10.1
    sink.close()
