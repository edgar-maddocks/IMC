import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf

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
data["previous_price3"] = data["mid_price"].shift(3)
data["previous_price2"] = data["mid_price"].shift(2)
data["previous_price1"] = data["mid_price"].shift()
data["next_price"] = data["mid_price"].shift(-1)
data = data.dropna()

X = data[
    [
        "previous_price3",
        "previous_price2",
        "previous_price1",
        "mid_price",
        "total_volume",
    ]
]
y = data["next_price"]

X = X.to_numpy()
y = y.to_numpy().reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, random_state=414, test_size=0.1
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="relu"),
    ]
)

optim = tf.keras.optimizers.Adam()
model.compile(optimizer=optim, loss=tf.keras.losses.MSE)

model.fit(x_train, y_train, epochs=25)

preds = model.predict(x_test)

print(preds)

print(mean_squared_error(y_test, preds))
print(type(model.layers[0].get_weights()[1]))
stuffs = {
    "layer_1_weights": model.layers[0].get_weights()[0].tolist(),
    "layer_1_biases": model.layers[0].get_weights()[1].tolist(),
    "layer_2_weights": model.layers[1].get_weights()[0].tolist(),
    "layer_2_biases": model.layers[1].get_weights()[1].tolist(),
    "final_layer_weights": model.layers[2].get_weights()[0].tolist(),
    "final_layer_biases": model.layers[2].get_weights()[1].tolist(),
}

with open("params.json", "w", encoding="utf-8") as fh:
    json.dump(stuffs, fh, ensure_ascii=False, indent=4)
