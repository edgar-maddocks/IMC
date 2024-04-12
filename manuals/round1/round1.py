import scipy
import numpy as np
import scipy.optimize
import numba
import time


@numba.jit(nopython=True)
def metric(args, random_reserves):
    profits = []
    for x in random_reserves:
        if args[0] > x:
            profits.append(1000 - args[0])
        elif args[1] > x:
            profits.append(1000 - args[1])
    return -sum(profits) / len(random_reserves)


@numba.jit(nopython=True)
def create_grid(bounds):
    tuples = []
    for i in range(bounds[0], bounds[1]):
        for j in range(bounds[0], bounds[1]):
            if i > j:
                continue
            else:
                tuples.append((i, j))

    return tuples


@numba.jit(nopython=True)
def grid_search():
    def cdf(x):
        return 100 * (np.sqrt(x) + 9)

    n_samples = 10000000
    rands = np.random.rand(n_samples)
    random_reserves = cdf(rands)
    grid = create_grid((900, 1000))
    vals = {}
    count = 0
    n_pairs = len(grid)
    for pair in grid:
        count += 1
        print(f"Simulating pair {count} / {n_pairs} on {n_samples} samples")
        vals[pair] = metric(pair, random_reserves)

    return vals


def scipyoptim():
    def cdf(x):
        return 100 * (np.sqrt(x) + 9)

    n_samples = 10000000
    rands = np.random.rand(n_samples)
    random_reserves = cdf(rands)
    result = (
        scipy.optimize.minimize(
            metric,
            [900, 1000],
            random_reserves,
            "Powell",
            bounds=[(900, 1000)],
            options={"disp": True},
        ),
    )

    return result


start = time.time()

"""vals = grid_search()
print("\n")
print(max(vals, key=lambda key: vals[key]))
print(max(vals.values()))
print("Time taken: ", time.time() - start)"""

result = scipyoptim()
print(result)
