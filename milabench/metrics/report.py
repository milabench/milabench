"""Filesystem-report aggregation helpers (no database dependency)."""

import numpy as np


def dropminmax(xs):
    xs = sorted(x for x in xs if x is not None)

    if len(xs) >= 5:
        xs = xs[1:-1]

    return xs


def sem(xs):
    xs = dropminmax(xs)
    return np.std(xs) / len(xs) ** 0.5


def min(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 0)


def q1(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 25)


def median(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 50)


def q3(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 75)


def max(xs):
    xs = dropminmax(xs)
    return np.percentile(xs, 100)


def mean(xs):
    xs = dropminmax(xs)
    return np.mean(xs)


def std(xs):
    xs = dropminmax(xs)
    return np.std(xs)


def count(xs):
    return len(xs)


def debug_count(xs):
    xs = dropminmax(xs)
    return len(xs)


def no_nan(fun):
    """NaN are not json serializable"""
    def wrapped(*args):
        return fun(*args)
    return fun


default_metrics = {
    "min": no_nan(min),
    "q1": no_nan(q1),
    "median": no_nan(median),
    "q3": no_nan(q3),
    "max": no_nan(max),
    "mean": no_nan(mean),
    "std": no_nan(std),
    "sem": no_nan(sem),
    "sum": np.sum,
    "count": no_nan(count),
    "debug_count": no_nan(debug_count),
}
