import time
import random


def try_with_random_exp_backoff(max_tries, a=0, b=1, c=2):
    delays = [a + b * (random.random() * c ** (i + 1)) for i in range(max_tries)]
    return try_with_backoff(delays)


def try_with_backoff(delays):
    def wrapper(func):
        def wrapped_func(*args, **kwargs):
            try_again = True
            last_exception = None
            for sleep_time in delays:
                try:
                    result = func(*args, **kwargs)
                    try_again = False
                    break
                except Exception as e:
                    last_exception = e
                    print(f"Encountered error: {e}")
                    print(f"Trying again in {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
            if try_again:
                raise last_exception
            return result

        return wrapped_func

    return wrapper


foo = None


# @try_with_backoff([1, 1, 1, 1, 1])
@try_with_random_exp_backoff(5)
def function_that_fails_first_n_calls(n):
    global foo
    foo = foo or 0
    time.sleep(1)
    if foo < n:
        foo += 1
        raise ValueError("foo")

    foo = 0
    return "bar"


def main():
    print(function_that_fails_first_n_calls(2))


if __name__ == "__main__":
    main()
