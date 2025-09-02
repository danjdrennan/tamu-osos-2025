"""_timer: A timer decorator to benchmark a function call."""

import functools
import timeit
from typing import Callable, Any


def timer(iterations: int = 100, repeats: int = 10) -> Callable:
    """Timer decorator that times function executions for multiple iterations.

    Args:
        iterations: Number of iterations to time the function for per repeat.
        repeats: The number of times to run the timing loop.

    Returns:
        Decorated function that returns a Result object with timing information
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            timing = timeit.repeat(
                lambda: func(*args, **kwargs),
                repeat=repeats,
                number=iterations,
            )

            return result, timing

        return wrapper

    return decorator


def main():
    import numpy as np

    @timer(iterations=1000, repeats=100)
    def mse(x: np.ndarray) -> np.ndarray:
        return np.mean(x**2)

    x = np.random.normal(size=90_000)
    result, timing = mse(x)
    print(result)
    print(f"{np.mean(timing):.5f} +/- {np.std(timing):.5f}")


if __name__ == "__main__":
    main()
