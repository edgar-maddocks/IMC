from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import List, Dict
import string

import numpy as np
import pandas as pd
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
            orders: List[Order] = []

            if len(product_order_depth.buy_orders) != 0:
                available_orders[product] = {"BID": product_order_depth.buy_orders}

            if len(product_order_depth.sell_orders) != 0:
                available_orders[product] = {"ASK": product_order_depth.sell_orders}

            result[product] = orders

        if state.timestamp == 0:
            trader_data_DICT = {}
            trader_data_DICT[state.timestamp] = available_orders
        elif state.timestamp != 0:
            trader_data_DICT = jsonpickle.loads(state.traderData)
            trader_data_DICT[state.timestamp] = available_orders
            if state.timestamp > (200 * 100):
                lowest_key = min([int(key) for key in trader_data_DICT.keys()])
                del trader_data_DICT[str(lowest_key)]

        serialized_trader_data_DICT = jsonpickle.dumps(trader_data_DICT)
        traderData = serialized_trader_data_DICT

        sma_spreads = {}

        for dict in list(trader_data_DICT.values()):
            print(dict)

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
