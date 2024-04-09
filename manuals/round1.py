import scipy
import numpy as np
import scipy.optimize


def cdf(x):
    return 100 * (np.sqrt(x) + 9)


def profit(bid):
    return 1000 - bid


def metric(args, random_reserves):
    profits = []
    for x in random_reserves:
        if args[0] > x:
            profits.append(1000 - args[0])
        elif args[1] > x:
            profits.append(1000 - args[1])
    return sum(profits) / len(random_reserves)


rands = np.random.rand(1000000)
random_reserves = cdf(rands)

print(metric((950, 970), random_reserves))
