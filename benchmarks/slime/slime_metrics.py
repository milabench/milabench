"""Collects metric events from the Ray workers so the driver can ``give`` them to voir."""

import ray

NAME = "milabench_metrics"
NAMESPACE = "milabench"


@ray.remote(num_cpus=0)
class Collector:
    def __init__(self):
        self.events = []

    def push(self, event):
        self.events.append(event)

    def drain(self):
        events, self.events = self.events, []
        return events


def create():
    return Collector.options(name=NAME, namespace=NAMESPACE).remote()


def get():
    return ray.get_actor(NAME, namespace=NAMESPACE)
