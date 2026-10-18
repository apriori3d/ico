"""Mergeable statistics of integer values: moments and a log histogram.

Memory is constant in the number of values, and two ``Stats`` of the same
quantity (from two threads or processes) merge exactly: count, sum, min,
max, mean and variance (Welford, merged with Chan's formula) and the
histogram (summed per bucket).

The histogram has 8 linear buckets per power of two (values below 8 have a
bucket each), like HDR Histogram: a quantile is the middle of its bucket,
within about 6 % of the true value, clamped to ``[min, max]``.
"""

from __future__ import annotations

import math
from typing import TypedDict, final

SUB_BITS = 3
SUB = 1 << SUB_BITS  # buckets per power of two


class StatsView(TypedDict):
    """Statistics of a metric, scaled to its unit; ``None`` while empty."""

    count: int
    mean: float | None
    std: float | None
    min: float | None
    max: float | None
    p50: float | None
    p95: float | None
    p99: float | None


def bucket(value: int) -> int:
    """Histogram bucket of a non-negative value."""
    if value < SUB:
        return max(value, 0)
    exponent = value.bit_length() - 1
    sub = (value >> (exponent - SUB_BITS)) & (SUB - 1)
    return (exponent - SUB_BITS + 1) * SUB + sub


def bucket_bounds(index: int) -> tuple[int, int]:
    """``[low, high)`` of a bucket."""
    if index < SUB:
        return index, index + 1
    exponent = index // SUB + SUB_BITS - 1
    width = 1 << (exponent - SUB_BITS)
    low = (SUB + index % SUB) * width
    return low, low + width


@final
class Stats:
    """Statistics of integer values (e.g. nanoseconds)."""

    __slots__ = ("count", "sum", "min", "max", "mean", "m2", "buckets")

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0
        self.min = 0
        self.max = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.buckets: dict[int, int] = {}

    def add(self, value: int) -> None:
        if self.count == 0:
            self.min = self.max = value
        elif value < self.min:
            self.min = value
        elif value > self.max:
            self.max = value
        self.count += 1
        self.sum += value
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        index = bucket(value)
        self.buckets[index] = self.buckets.get(index, 0) + 1

    def merge(self, other: Stats) -> None:
        """Add the values of ``other``."""
        if other.count == 0:
            return
        if self.count == 0:
            self.min, self.max = other.min, other.max
        else:
            self.min = min(self.min, other.min)
            self.max = max(self.max, other.max)
        count = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / count
        self.mean += delta * other.count / count
        self.count = count
        self.sum += other.sum
        for index, n in other.buckets.items():
            self.buckets[index] = self.buckets.get(index, 0) + n

    def copy(self) -> Stats:
        out = Stats()
        out.merge(self)
        return out

    def quantile(self, q: float) -> float | None:
        """The ``q``-quantile (nearest rank), from the histogram."""
        if self.count == 0:
            return None
        rank = max(1, math.ceil(q * self.count))
        seen = 0
        for index in sorted(self.buckets):
            seen += self.buckets[index]
            if seen >= rank:
                low, high = bucket_bounds(index)
                middle = (low + high - 1) / 2
                return min(max(middle, self.min), self.max)
        return float(self.max)

    def view(self, scale: float) -> StatsView:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "p50": None,
                "p95": None,
                "p99": None,
            }

        def scaled(value: float | None) -> float | None:
            return None if value is None else value * scale

        return {
            "count": self.count,
            "mean": self.mean * scale,
            "std": math.sqrt(self.m2 / self.count) * scale,
            "min": self.min * scale,
            "max": self.max * scale,
            "p50": scaled(self.quantile(0.5)),
            "p95": scaled(self.quantile(0.95)),
            "p99": scaled(self.quantile(0.99)),
        }
