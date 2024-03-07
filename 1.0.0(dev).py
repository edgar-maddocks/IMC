from datamodel import OrderDepth, UserId, TradingState, Order, Trade
from typing import List, Dict
import string

import numpy as np
import pandas as pd
import statistics
import jsonpickle
import collections


class Trader:

    position = {"STARFRUIT" : 0, "AMETHYSTS": 0}
    POSITION_LIMITS = {"STARFRUIT" : 20, "AMETHYSTS" : 20}

    def generate_bid_ask_prices(self, mid_price: float, spread: float, position_weight: float):
        ask_price = mid_price + ((spread * mid_price) * position_weight)
        bid_price = mid_price - ((spread * mid_price) * position_weight)

        return bid_price, ask_price

    def calculate_mid_price(self, best_ask, best_bid):
        return (best_ask + best_bid) / 2
    
    def get_bb_ba(self, state: TradingState, product: str):
        result = []
        if len(state.order_depths[product].buy_orders) != 0:
            bids = sorted(list(state.order_depths[product].buy_orders.keys()), reverse=True)
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
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMITS[product]:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMITS[product] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMITS[product]:
                order_for = max(-vol, -self.POSITION_LIMITS[product] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        return orders
    
    def calculate_position_weights(self):
        total_pos = sum([abs(x) for x in list(self.position.values())])
        position_weights = {"STARFRUIT" : 0, "AMETHYSTS": 0}
        if total_pos == 0:
            for key in position_weights.keys():
                position_weights[key] = 1
        else:
            for key in position_weights.keys():
                position_weights[key] = (total_pos / self.position[key]) - 1

        return position_weights

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
            if state.timestamp > (20 * 100):
                lowest_key = min([int(key) for key in trader_data_DICT.keys()])
                del trader_data_DICT[str(lowest_key)]

        serialized_trader_data_DICT = jsonpickle.dumps(trader_data_DICT)
        traderData = serialized_trader_data_DICT

        position_weights = self.calculate_position_weights()
        spreads = self.calculate_spreads(trader_data_DICT, products)
        for product in products:
            bb, ba = self.get_bb_ba(state, product)
            mid_price = self.calculate_mid_price(ba, bb)
            acc_bid, acc_ask = self.generate_bid_ask_prices(mid_price, 0.025, position_weights[product])
            result[product] = self.compute_orders(product, state.order_depths[product], acc_bid, acc_ask)

            print("ACC BID:", acc_bid)
            print("BEST BID:", ba)

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData
