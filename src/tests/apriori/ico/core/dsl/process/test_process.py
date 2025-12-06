from apriori.ico.core.meta.flow_meta import IcoFlowMeta
from apriori.ico.core.operator import IcoOperator
from apriori.ico.core.process import IcoProcess


def test_process_applies_body_multiple_times() -> None:
    """
    Test that IcoProcess applies its body to the context multiple times.
    """

    # A stateful step: increment context by 1
    increment = IcoOperator[int, int](lambda x: x + 1, name="increment")

    # Create process: apply increment 3 times
    process = IcoProcess[int](body=increment, num_iterations=3)

    # Execute process starting with context = 0
    result = process(0)

    # Should apply increment 3 times: 0 → 1 → 2 → 3
    assert result == 3


def test_process_body_can_mutate_context() -> None:
    """
    Test that process body can mutate the context in-place (stateful behavior).
    """

    class Counter:
        def __init__(self) -> None:
            self.value = 0

        def step(self) -> "Counter":
            self.value += 1
            return self

    # Wrap stateful callable
    step = IcoOperator[Counter, Counter](lambda c: c.step(), name="step")

    process = IcoProcess[Counter](body=step, num_iterations=5)

    c = Counter()
    result = process(c)

    # Context should be mutated in-place
    assert result.value == 5
    assert c is result  # same object reference


def test_process_structure_representation() -> None:
    """
    Test that IcoProcess exposes correct flow structure.
    """

    step = IcoOperator[int, int](lambda x: x * 2, name="scale")
    process = IcoProcess[int](body=step, num_iterations=2)

    flow = IcoFlowMeta.from_node(process)

    # Root node should be a process
    assert flow.node_type == "process"
    assert len(flow.children) == 1
    assert flow.children[0].name == "scale"
    assert flow.children[0].node_type == "operator"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
