from ico import IcoProcess, operator

Context = tuple[int, int]


@operator()
def fib_step(state: Context) -> Context:
    a, b = state
    return (b, a + b)


@operator()
def first(state: Context) -> int:
    return state[0]


fib8 = IcoProcess(fib_step, num_iterations=8) | first

print(f"{fib8((0, 1))=}")  # 21

fib8.describe()
