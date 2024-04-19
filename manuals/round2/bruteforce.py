import itertools
import time

start = time.time()

conversions = {
    "P": {"P": 1, "W": 0.48, "S": 1.52, "C": 0.71},
    "W": {"P": 2.05, "W": 1, "S": 3.26, "C": 1.56},
    "S": {"P": 0.64, "W": 0.3, "S": 1, "C": 0.46},
    "C": {"P": 1.41, "W": 0.61, "S": 2.08, "C": 1},
}

mx_score = (-1, "")
for combo in itertools.product(["P", "W", "S", "C"], repeat=4):
    score = 1
    combo = ("C", *combo, "C")
    for i, product in enumerate(combo[1:]):
        score *= conversions[combo[i]][product]

    if score > mx_score[0]:
        mx_score = (score, combo)

print("FINAL MULTIPLIER: ", mx_score[0])
print("FINAL PATH: ", mx_score[1][1:-1])
print("TIME ELAPSED: ", time.time() - start)
