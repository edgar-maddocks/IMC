import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("starfruit_test.csv", sep=";")
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
data["sma50"] = data["mid_price"].rolling(50).mean()
data["sma5"] = data["mid_price"].rolling(25).mean()
data["previous_price1"] = data["mid_price"].shift()
data["next_price"] = data["mid_price"].shift(-1)
data = data.dropna()
print(data)

fig, ax = plt.subplots(1, 1, figsize=(30, 10))
ax.set_xlim(0, 1000 * 100)
ax.plot(data.index, data["mid_price"])
ax.plot(data.index, data["sma50"], c="g")
ax.plot(data.index, data["sma5"], c="r")

plt.show()

# X = data[["previous_price1", "mid_price", "total_volume"]]
# y = data["next_price"]

# X = X.to_numpy()
# y = y.to_numpy().reshape(-1, 1)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# model = LinearRegression()

# model.fit(x_train, y_train)
# preds = model.predict(x_test)

# print("R2:", r2_score(y_test, preds))
# print("MSE:", mean_squared_error(y_test, preds))
# print("##############################")
# print(model.coef_)
# print(model.intercept_)
