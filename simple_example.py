from ico.core.process import IcoProcess


# Fibonacci as an iterative process
def fib_step(state: tuple[int, int]) -> tuple[int, int]:
    return (state[1], state[0] + state[1])


fib_process = IcoProcess(fib_step, num_iterations=8, name="Fibonacci Process")


# result = fib_process((0, 1))
# print(result)  # (21, 34) - 8th Fibonacci number

fib_process.describe()
