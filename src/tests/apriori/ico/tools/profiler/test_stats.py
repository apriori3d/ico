import math
import random
import statistics

from ico.tools.profiler.stats import Stats, bucket, bucket_bounds


def close(a: float | None, b: float, rel: float = 1e-9) -> bool:
    return a is not None and math.isclose(a, b, rel_tol=rel)


def stats_of(values: list[int]) -> Stats:
    stats = Stats()
    for value in values:
        stats.add(value)
    return stats


def test_buckets_cover_values_without_gaps() -> None:
    previous_high = 0
    for index in range(200):
        low, high = bucket_bounds(index)
        assert low == previous_high
        assert bucket(low) == bucket(high - 1) == index
        previous_high = high


def test_moments_and_quantiles() -> None:
    rng = random.Random(1)
    values = [int(rng.lognormvariate(14, 1)) for _ in range(5000)]
    stats = stats_of(values)
    view = stats.view(1.0)
    assert view["count"] == 5000
    assert close(view["mean"], statistics.fmean(values))
    assert close(view["std"], statistics.pstdev(values))
    assert (view["min"], view["max"]) == (min(values), max(values))
    ordered = sorted(values)
    for q, estimate in ((0.5, view["p50"]), (0.95, view["p95"]), (0.99, view["p99"])):
        exact = ordered[int(q * len(values)) - 1]
        assert close(estimate, exact, rel=0.07)


def test_merge_equals_one_pass() -> None:
    rng = random.Random(2)
    values = [rng.randrange(1, 10**9) for _ in range(3000)]
    merged = stats_of(values[:1000])
    merged.merge(stats_of(values[1000:]))
    merged.merge(Stats())
    whole = stats_of(values)
    a, b = merged.view(1.0), whole.view(1.0)
    assert a["count"] == b["count"]
    assert (a["min"], a["max"], a["p50"], a["p99"]) == (
        b["min"],
        b["max"],
        b["p50"],
        b["p99"],
    )
    assert b["mean"] is not None and close(a["mean"], b["mean"])
    assert b["std"] is not None and close(a["std"], b["std"])
    assert merged.buckets == whole.buckets
    assert merged.sum == whole.sum


def test_empty() -> None:
    view = Stats().view(1e-9)
    assert view["count"] == 0
    assert view["mean"] is None and view["p99"] is None
