from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import List, Dict
import string

import numpy as np
import pandas as pd
import statistics
import jsonpickle
import collections


class Trader:

    position = {"STARFRUIT": 0, "AMETHYSTS": 0}
    POSITION_LIMITS = {"STARFRUIT": 20, "AMETHYSTS": 20}

    def calc_next_star_mid(self, price_inputs):
        price_weights = [
            0.09497349,
            0.10797433,
            0.11837098,
            0.15644577,
            0.22009514,
            0.30172641,
        ]
        intercept = 2.09558459
        next_price = intercept

        for i, val in enumerate(price_inputs):
            next_price += val * price_weights[i]

        return int(round(next_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask

        return tot_vol, best_val

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True)
        )

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if (
                (ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))
            ) and cpos < self.POSITION_LIMITS[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMITS[product] - cpos)
                cpos += order_for
                assert order_for >= 0
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(
            undercut_buy, acc_bid - 1
        )  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.POSITION_LIMITS[product]) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMITS[product] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMITS[product]) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMITS[product] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.POSITION_LIMITS[product]:
            num = min(40, self.POSITION_LIMITS[product] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if (
                (bid > acc_ask) or ((self.position[product] > 0) and (bid == acc_ask))
            ) and cpos > -self.POSITION_LIMITS[product]:
                order_for = max(-vol, -self.POSITION_LIMITS[product] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert order_for <= 0
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMITS[product]) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMITS[product] - cpos)
            orders.append(Order(product, max(undercut_sell - 1, acc_ask + 1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMITS[product]) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMITS[product] - cpos)
            orders.append(Order(product, max(undercut_sell + 1, acc_ask + 1), num))
            cpos += num

        if cpos > -self.POSITION_LIMITS[product]:
            num = max(-40, -self.POSITION_LIMITS[product] - cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True)
        )

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if (
                (ask <= acc_bid)
                or ((self.position[product] < 0) and (ask == acc_bid + 1))
            ) and cpos < self.POSITION_LIMITS[product]:
                order_for = min(-vol, self.POSITION_LIMITS[product] - cpos)
                cpos += order_for
                assert order_for >= 0
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(
            undercut_buy, acc_bid
        )  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < self.POSITION_LIMITS[product]:
            num = self.POSITION_LIMITS[product] - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if (
                (bid >= acc_ask)
                or ((self.position[product] > 0) and (bid + 1 == acc_ask))
            ) and cpos > -self.POSITION_LIMITS[product]:
                order_for = max(-vol, -self.POSITION_LIMITS[product] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert order_for <= 0
                orders.append(Order(product, bid, order_for))

        if cpos > -self.POSITION_LIMITS[product]:
            num = -self.POSITION_LIMITS[product] - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def get_past_prices(self, trader_data_DICT, product):
        prices = {}
        bids = np.array([])
        asks = np.array([])
        for dict in list(trader_data_DICT.values()):
            if len(dict[product].keys()) > 0:
                bids = np.append(
                    bids,
                    np.max([float(x) for x in list(dict[product]["BID"].keys())]),
                )
                prices["BID"] = bids
            if len(dict[product].keys()) > 0:
                asks = np.append(
                    asks,
                    np.min([float(x) for x in list(dict[product]["ASK"].keys())]),
                )
                prices["ASK"] = asks

        return prices

    def run(self, state: TradingState):
        for product, pos in state.position.items():
            self.position[product] = pos
        # Orders to be placed on exchange matching engine
        result = {}
        products = []
        available_orders = {}
        for product in state.order_depths:
            products.append(product)
            product_order_depth: OrderDepth = state.order_depths[product]

            available_orders[product] = {}
            if len(product_order_depth.buy_orders) != 0:
                available_orders[product]["BID"] = product_order_depth.buy_orders

            if len(product_order_depth.sell_orders) != 0:
                available_orders[product]["ASK"] = product_order_depth.sell_orders

        if state.timestamp == 0:
            trader_data_DICT = {}
            trader_data_DICT[state.timestamp] = available_orders
        elif state.timestamp != 0:
            trader_data_DICT = jsonpickle.loads(state.traderData)
            trader_data_DICT[state.timestamp] = available_orders
            if state.timestamp > (6 * 100):
                lowest_key = min([int(key) for key in trader_data_DICT.keys()])
                del trader_data_DICT[str(lowest_key)]

        serialized_trader_data_DICT = jsonpickle.dumps(trader_data_DICT)
        traderData = serialized_trader_data_DICT

        for product in products:
            if product == "AMETHYSTS":
                acc_bid = 10000
                acc_ask = 10000

                result[product] = self.compute_orders(
                    product, state.order_depths[product], acc_bid, acc_ask
                )
            if product == "STARFRUIT" and state.timestamp > (6 * 100):
                product_prices = self.get_past_prices(trader_data_DICT, product)
                mid_prices = (product_prices["BID"] + product_prices["ASK"]) / 2
                next_price = self.calc_next_star_mid(mid_prices[-6:].tolist())
                print("MID PRICE:", mid_prices[-1])
                print("PRED PRICE:", next_price)

                result[product] = self.compute_orders_regression(
                    product, state.order_depths[product], next_price - 1, next_price + 1
                )

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
