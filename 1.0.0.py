from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import List, Dict
import string

import numpy as np
import pandas as pd
import statistics
import jsonpickle


class Trader:

    def run(self, state: TradingState):

        # Orders to be placed on exchange matching engine
        result = {}
        products = []
        available_orders = {}
        for product in state.order_depths:
            products.append(product)
            product_order_depth: OrderDepth = state.order_depths[product]
            my_orders: List[Order] = []

            available_orders[product] = {}
            if len(product_order_depth.buy_orders) != 0:
                available_orders[product]["BID"] = product_order_depth.buy_orders

            if len(product_order_depth.sell_orders) != 0:
                available_orders[product]["ASK"] = product_order_depth.sell_orders

            result[product] = my_orders

        if state.timestamp == 0:
            trader_data_DICT = {}
            trader_data_DICT[state.timestamp] = available_orders
        elif state.timestamp != 0:
            trader_data_DICT = jsonpickle.loads(state.traderData)
            trader_data_DICT[state.timestamp] = available_orders
            if state.timestamp > (20 * 100):
                lowest_key = min([int(key) for key in trader_data_DICT.keys()])
                del trader_data_DICT[str(lowest_key)]

        serialized_trader_data_DICT = jsonpickle.dumps(trader_data_DICT)
        traderData = serialized_trader_data_DICT

        spreads = {}
        for product in products:
            bids = np.array([])
            asks = np.array([])
            for dict in list(trader_data_DICT.values()):
                if len(dict[product].keys()) > 0:
                    bids = np.append(
                        bids,
                        np.mean([float(x) for x in list(dict[product]["BID"].keys())]),
                    )
                if len(dict[product].keys()) > 0:
                    asks = np.append(
                        asks,
                        np.mean([float(x) for x in list(dict[product]["ASK"].keys())]),
                    )
            spread = asks - bids
            spreads[product] = spread

        print(spreads)

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
