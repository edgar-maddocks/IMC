import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

data = pd.read_csv("starfruit strat/starfruit_test.csv", sep=";")
data = data[data["product"] == "STARFRUIT"]
data = data.drop("day", axis=1)
data["total_volume"] = (
    data["bid_volume_1"].fillna(0)
    + data["bid_volume_2"].fillna(0)
    + data["bid_volume_3"].fillna(0)
    + data["ask_volume_1"].fillna(0)
    + data["ask_volume_2"].fillna(0)
    + data["ask_volume_3"].fillna(0)
)
data = data[["mid_price", "timestamp", "total_volume"]]
data = data.set_index("timestamp")
data["previous_price3"] = data["mid_price"].shift(3)
data["previous_price2"] = data["mid_price"].shift(2)
data["previous_price1"] = data["mid_price"].shift()
data["next_price"] = data["mid_price"].shift(-1)
data = data.dropna()

autocorrelation_plot(data["mid_price"])
plt.show()
