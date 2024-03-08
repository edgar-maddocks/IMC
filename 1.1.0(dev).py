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
    first_signal = {"STARFRUIT": False, "AMETHYSTS": False}
    short_sma_above = {"STARTFRUIT": False, "AMETHYSTS": False}

    def calculate_mid_price(self, best_ask, best_bid):
        return (best_ask + best_bid) / 2

    def get_bb_ba(self, state: TradingState, product: str):
        result = []
        if len(state.order_depths[product].buy_orders) != 0:
            bids = sorted(
                list(state.order_depths[product].buy_orders.keys()), reverse=True
            )
            result.append(bids[0])
        else:
            result.append(None)
        if len(state.order_depths[product].sell_orders) != 0:
            asks = sorted(list(state.order_depths[product].sell_orders.keys()))
            result.append(asks[0])
        else:
            result.append(None)
        return result[0], result[1]

    def calculate_spreads(self, trader_data_DICT, products):
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

        return spreads

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

        return orders

    def get_past_prices(self, trader_data_DICT, product):
        prices = {}
        bids = np.array([])
        asks = np.array([])
        for dict in list(trader_data_DICT.values()):
            if len(dict[product].keys()) > 0:
                bids = np.append(
                    bids,
                    np.mean([float(x) for x in list(dict[product]["BID"].keys())]),
                )
                prices["BID"] = bids
            if len(dict[product].keys()) > 0:
                asks = np.append(
                    asks,
                    np.mean([float(x) for x in list(dict[product]["ASK"].keys())]),
                )
                prices["ASK"] = asks

        return prices

    def fill_orders(self, product, order_depth, order_vol, buy=0):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(
            sorted(order_depth.buy_orders.items(), reverse=True)
        )

        cpos = self.position[product]

        if buy == 0:
            available_vol = 0
            for bid, vol in obuy.items():
                available_vol += abs(vol)
                if available_vol < order_vol:
                    continue
                else:
                    if cpos > 0:
                        order_vol += cpos
                    orders.append(Order(product, bid, -order_vol))

        if buy == 1:
            available_vol = 0
            for ask, vol in osell.items():
                available_vol += abs(vol)
                if available_vol < order_vol:
                    continue
                else:
                    if cpos < 0:
                        order_vol -= cpos
                    orders.append(Order(product, ask, -order_vol))

        return orders

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
            if state.timestamp > (50 * 100):
                lowest_key = min([int(key) for key in trader_data_DICT.keys()])
                del trader_data_DICT[str(lowest_key)]

        serialized_trader_data_DICT = jsonpickle.dumps(trader_data_DICT)
        traderData = serialized_trader_data_DICT

        if state.timestamp < (50 * 100):
            pass
        else:
            for product in products:
                product_prices = self.get_past_prices(trader_data_DICT, product)
                mid_prices = (product_prices["BID"] + product_prices["ASK"]) / 2
                short_sma = np.mean(mid_prices[-int(len(mid_prices) / 3) :])
                long_sma = np.mean(mid_prices)

                total_sell_vol, _ = self.values_extract(
                    state.order_depths[product].sell_orders
                )
                total_buy_vol, _ = self.values_extract(
                    state.order_depths[product].sell_orders
                )
                if self.first_signal[product] is False:
                    self.short_sma_above[product] = (
                        True if short_sma > long_sma else False
                    )
                    self.first_signal[product] = True
                elif short_sma > long_sma and self.short_sma_above[product] is False:
                    result[product] = self.fill_orders(
                        product,
                        state.order_depths[product],
                        int(total_sell_vol / 2),
                        buy=1,
                    )
                    print("BUYING", product)
                    self.short_sma_above[product] = True
                elif short_sma < long_sma and self.short_sma_above[product] is True:
                    result[product] = self.fill_orders(
                        product,
                        state.order_depths[product],
                        int(total_buy_vol / 2),
                    )
                    print("SELLING", product)
                    self.short_sma_above[product] = False

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
