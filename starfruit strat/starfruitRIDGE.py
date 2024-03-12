from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

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

data["previous_price5"] = data["mid_price"].shift(5)
data["previous_price4"] = data["mid_price"].shift(4)
data["previous_price3"] = data["mid_price"].shift(3)
data["previous_price2"] = data["mid_price"].shift(2)
data["previous_price1"] = data["mid_price"].shift()
data["next_price"] = data["mid_price"].shift(-1)
data = data.dropna()

X = data[["previous_price5", "previous_price4", "previous_price3", "previous_price2", "previous_price1", "mid_price"]]
y = data["next_price"]

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=414, test_size=0.1
)


model = Ridge()
model.fit(x_train, y_train)

preds = model.predict(x_test)
print(model.coef_)
print(model.intercept_)
print(mean_squared_error(preds, y_test))
