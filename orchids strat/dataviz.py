day_0 = r"C:\Users\edgar\OneDrive\Documents\Code\OptRLResearch\IMC\orchids strat\round-2-island-data-bottle (1)\prices_round_2_day_-1.csv"
day_1 = r"C:\Users\edgar\OneDrive\Documents\Code\OptRLResearch\IMC\orchids strat\round-2-island-data-bottle (1)\prices_round_2_day_0.csv"
day_2 = r"C:\Users\edgar\OneDrive\Documents\Code\OptRLResearch\IMC\orchids strat\round-2-island-data-bottle (1)\prices_round_2_day_1.csv"

import pandas as pd

day0 = pd.read_csv(day_0, sep=";")
day1 = pd.read_csv(day_1, sep=";")
day2 = pd.read_csv(day_2, sep=";")

day1["timestamp"] += 1000000
day2["timestamp"] += 2000000

data = pd.concat([day0, day1, day2])
data = data.set_index("timestamp")


data = data[["ORCHIDS", "SUNLIGHT"]]

for i in range(5, len(data) - 5):
    if (
        data["SUNLIGHT"].iloc[i] / 360 < 7
        and data["SUNLIGHT"].iloc[i + 1] / 360 < 7
        and data["SUNLIGHT"].iloc[i + 2] / 360 < 7
        and data["SUNLIGHT"].iloc[i + 3] / 360 > data["SUNLIGHT"].iloc[i + 2] / 360
        and data["SUNLIGHT"].iloc[i + 4] / 360 > data["SUNLIGHT"].iloc[i + 3] / 360
    ):
        print("BUYING AT", data.iloc[i])
