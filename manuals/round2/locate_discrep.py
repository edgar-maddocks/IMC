exchange_mat = [
    [1, 0.48, 1.52, 0.71],
    [2.05, 1, 3.26, 1.56],
    [0.64, 0.3, 1, 0.46],
    [1.41, 0.61, 2.08, 1],
]

import numpy as np


class ArbitrageGraph:
    def __init__(self, exchange_mat):
        self.exchange_mat = exchange_mat
        self.n_products = len(exchange_mat)
        self.log_graph = -np.log(np.array(exchange_mat))

    def __repr__(self):
        return str(self.log_graph)

    def bellman_ford(self, src):

        # Step 1: Initialize distances from src to all other vertices
        # as INFINITE
        dist = [float("Inf")] * self.n_products
        dist[src] = 0


print(ArbitrageGraph(exchange_mat))
