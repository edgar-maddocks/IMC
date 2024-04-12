import itertools

conversions = {
    "P": {"P": 1, "W": 0.48, "S": 1.52, "C": 0.71},
    "W": {"P": 2.05, "W": 1, "S": 3.26, "C": 1.56},
    "S": {"P": 0.64, "W": 0.3, "S": 1, "C": 0.46},
    "C": {"P": 1.41, "W": 0.61, "S": 2.08, "C": 1},
}


def foo(l):
    for i in itertools.product(*[l] * 4):
        yield i


combos = [x for x in foo(["P", "W", "S", "C"])]
outputs = {}
for combo in combos:
    combo = ("C", *combo, "C")
    init = 2000000
    count = 0
    for i, val in enumerate(combo):
        init *= conversions[combo[i - 1]][val]

    outputs[combo] = init

opt_combo = max(outputs, key=lambda key: outputs[key])
print(opt_combo)
print(outputs[opt_combo])
