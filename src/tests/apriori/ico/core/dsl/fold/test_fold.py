"""IcoFold: a fold over a source with an ordinary step over (item, state)."""

from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from ico import IcoOperator, IcoSource, plan_view, validate_signatures
from ico.core.fold import IcoFold
from ico.core.runtime.cancel import IcoCancelled, IcoCancelToken, cancellable


@dataclass(frozen=True)
class Stats:
    total: int = 0
    count: int = 0


def numbers() -> Iterator[int]:
    return iter([1, 2, 3, 4])


def add(pair: tuple[int, Stats]) -> Stats:
    number, stats = pair
    return Stats(stats.total + number, stats.count + 1)


def mean(stats: Stats) -> Stats:
    return stats


def test_a_fold_threads_the_state_through_the_items() -> None:
    fold = IcoFold(IcoSource(numbers), add)
    assert fold(Stats()) == Stats(total=10, count=4)
    assert fold(Stats(total=100)) == Stats(total=110, count=4)  # starts from its input
    assert fold.signature.format() == "Stats → Stats"


def test_a_step_that_does_not_take_the_items_is_reported() -> None:
    def words() -> Iterator[str]:
        return iter(["a", "b"])

    assert validate_signatures(IcoFold(IcoSource(numbers), add)) == []
    wrong = IcoFold(IcoSource(words), add)  # pyright: ignore[reportArgumentType]
    [issue] = validate_signatures(wrong)
    assert issue.kind == "mismatch"
    assert issue.output == tuple[str, Stats] and issue.input == tuple[int, Stats]


def test_a_cancelled_fold_stops_between_items() -> None:
    token = IcoCancelToken()
    seen: list[int] = []

    def count(pair: tuple[int, Stats]) -> Stats:
        seen.append(pair[0])
        if pair[0] == 2:
            token.cancel("enough")
        return add(pair)

    with cancellable(token), pytest.raises(IcoCancelled, match="enough"):
        IcoFold(IcoSource(numbers), count)(Stats())
    assert seen == [1, 2]


def test_the_plan_shows_the_source_the_step_and_what_it_emits() -> None:
    fold = IcoFold(IcoSource(numbers), IcoOperator(add) | IcoOperator(mean), name="sum")
    rows = ["".join(s["text"] for s in row[0]) for row in plan_view(fold)["rows"]]
    assert rows == [
        "🧠IcoFold()",
        "╭── for each in ",
        "│   📚IcoSource(numbers)",
        "├─▸ fold",
        "│   add",
        "│   mean",
        "╰─▸ emit",
    ]
