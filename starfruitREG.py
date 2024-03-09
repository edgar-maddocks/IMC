import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("starfruit_test.csv", sep=";")
data = data[data["product"] == "STARFRUIT"]
data = data.drop("day", axis=1)
data = data[["mid_price", "timestamp"]]
data = data.set_index("timestamp")
data["previous_price"] = data["mid_price"].shift()
data["next_price"] = data["mid_price"].shift(-1)
data = data.dropna()


X = data[["previous_price", "mid_price"]]
y = data["next_price"]

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LinearRegression()

model.fit(x_train, y_train)
preds = model.predict(x_test)

print("R2:", r2_score(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
print("##############################")
print(model.coef_)
print(model.intercept_)
