from datamodel import (
    OrderDepth,
    UserId,
    TradingState,
    Order,
    Trade,
    Symbol,
    ProsperityEncoder,
    Observation,
    Listing,
)
from typing import List, Dict, Any
import json

import numpy as np
import jsonpickle
import collections

from datamodel import (
    OrderDepth,
    UserId,
    TradingState,
    Order,
    Trade,
    Symbol,
    ProsperityEncoder,
    Observation,
    Listing,
)
from typing import List, Dict, Any
import json

import numpy as np
from statistics import NormalDist
import jsonpickle
import collections
import math as mt


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append(
                [listing["symbol"], listing["product"], listing["denomination"]]
            )

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:

    position = {
        "STARFRUIT": 0,
        "AMETHYSTS": 0,
        "ORCHIDS": 0,
        "CHOCOLATE": 0,
        "STRAWBERRIES": 0,
        "ROSES": 0,
        "GIFT_BASKET": 0,
        "COCONUT": 0,
        "COCONUT_COUPON": 0,
    }
    POSITION_LIMITS = {
        "STARFRUIT": 20,
        "AMETHYSTS": 20,
        "ORCHIDS": 100,
        "CHOCOLATE": 250,
        "STRAWBERRIES": 350,
        "ROSES": 60,
        "GIFT_BASKET": 60,
        "COCONUT": 300,
        "COCONUT_COUPON": 600,
    }
    starfrut_signal = None

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

    def compute_gb_orders(self, product, state, mids, component_basket_price):
        orders = []
        cpos = state.position.get(product, 0)
        bid_price, bid_amount = list(state.order_depths[product].buy_orders.items())[0]
        ask_price, ask_amount = list(state.order_depths[product].sell_orders.items())[0]
        if mids["GIFT_BASKET"] - component_basket_price > 435.2:
            orders.append(
                Order(
                    product,
                    bid_price,
                    max(-self.POSITION_LIMITS[product] - cpos, -bid_amount),
                )
            )
            orders.append(Order(product, bid_price, -4))

        elif mids["GIFT_BASKET"] - component_basket_price < 339.8:  # 337.8
            orders.append(
                Order(
                    product,
                    ask_price,
                    min(self.POSITION_LIMITS[product] - cpos, -ask_amount),
                )
            )
            orders.append(Order(product, ask_price, 4))

        return orders

    def compute_copy_trader(self, state, product, persona):
        orders = []
        if product in list(state.market_trades.keys()):
            for trade in state.market_trades[product]:
                order_depth = state.order_depths[product]
                if trade.seller == persona:
                    if len(list(order_depth.buy_orders.items())) > 0:
                        best_bid, best_bid_amount = list(
                            order_depth.buy_orders.items()
                        )[0]
                    orders.append(
                        Order(
                            product,
                            best_bid,
                            max(
                                -self.POSITION_LIMITS[product]
                                - state.position.get(product, 0),
                                -best_bid_amount,
                            ),
                        )
                    )
                elif trade.buyer == persona:
                    if len(list(order_depth.sell_orders.items())) > 0:
                        best_ask, best_ask_amount = list(
                            order_depth.sell_orders.items()
                        )[0]
                    orders.append(
                        Order(
                            product,
                            best_ask,
                            min(
                                self.POSITION_LIMITS[product]
                                - state.position.get(product, 0),
                                -best_ask_amount,
                            ),
                        )
                    )
                else:
                    if product in list(state.own_trades.keys()):
                        if (
                            state.own_trades[product][-1].timestamp
                            == (state.timestamp - 100)
                            and abs(state.position.get(product))
                            != self.POSITION_LIMITS[product]
                        ):
                            if state.own_trades[product][-1].quantity < 0:
                                if len(list(order_depth.buy_orders.items())) > 0:
                                    best_bid, best_bid_amount = list(
                                        order_depth.buy_orders.items()
                                    )[0]
                                orders.append(
                                    Order(
                                        product,
                                        best_bid,
                                        max(
                                            -self.POSITION_LIMITS[product]
                                            - state.position.get(product, 0),
                                            -best_bid_amount,
                                        ),
                                    )
                                )
                            elif state.own_trades[product][-1].quantity > 0:
                                if len(list(order_depth.sell_orders.items())) > 0:
                                    best_ask, best_ask_amount = list(
                                        order_depth.sell_orders.items()
                                    )[0]
                                orders.append(
                                    Order(
                                        product,
                                        best_ask,
                                        min(
                                            self.POSITION_LIMITS[product]
                                            - state.position.get(product, 0),
                                            -best_ask_amount,
                                        ),
                                    )
                                )

        return orders

    def black_scholes_price(
        self, S, K=10000, t=250, r=0, sigma=0.010063617743242183, option_type="call"
    ):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        price = S * NormalDist().cdf(d1) - K * np.exp(-r * t) * NormalDist().cdf(d2)
        return price

    def run(self, state: TradingState):
        products = ["AMETHYSTS", "STARFRUIT", "ORCHIDS", "GIFT_BASKET", "COCONUT"]
        result = {}
        conversions = 0

        for product, pos in state.position.items():
            self.position[product] = pos

        data = {}
        for product in state.order_depths:
            if product == "STARFRUIT" or "COCONUT":
                product_order_depth: OrderDepth = state.order_depths[product]

                best_ask, best_bid = (0, 0)
                if len(product_order_depth.buy_orders) != 0:
                    best_bid = np.max(
                        [float(x) for x in list(product_order_depth.buy_orders.keys())]
                    )

                if len(product_order_depth.sell_orders) != 0:
                    best_ask = np.min(
                        [float(x) for x in list(product_order_depth.sell_orders.keys())]
                    )

                data[product] = (best_bid + best_ask) / 2
        if state.timestamp == 0:
            trader_data_DICT = {}
            trader_data_DICT[state.timestamp] = data
        elif state.timestamp != 0:
            trader_data_DICT = jsonpickle.loads(state.traderData)
            trader_data_DICT[state.timestamp] = data
            if state.timestamp > (6 * 100):
                lowest_key = min([int(key) for key in trader_data_DICT.keys()])
                del trader_data_DICT[str(lowest_key)]

        serialized_trader_data_DICT = jsonpickle.dumps(trader_data_DICT)

        for product in products:
            result[product] = []
            if product == "AMETHYSTS":
                acc_bid = 10000
                acc_ask = 10000

                result[product] = self.compute_orders(
                    product, state.order_depths[product], acc_bid, acc_ask
                )
            if product == "STARFRUIT" and state.timestamp > (6 * 100):
                mid_prices = [x["STARFRUIT"] for x in list(trader_data_DICT.values())]
                next_price = self.calc_next_star_mid(mid_prices[-6:])

                result[product] = self.compute_orders_regression(
                    product, state.order_depths[product], next_price - 1, next_price + 1
                )
            if product == "ORCHIDS":
                ## MARKET MAKE IN OWN ISLAND
                ## NEUTRALIZE POSITION USING CONVERSIONS

                conversionObs = state.observations.conversionObservations[product]
                fair_ask = (
                    conversionObs.askPrice
                    + conversionObs.importTariff
                    + conversionObs.transportFees
                ) + 1
                fair_bid = (
                    conversionObs.bidPrice
                    - conversionObs.transportFees
                    - conversionObs.exportTariff
                ) - 1

                result[product].append(Order(product, round(fair_ask), -100))
                result[product].append(Order(product, round(fair_bid), 100))
                conversions = -state.position.get(product, 0)
            if product == "GIFT_BASKET":
                mids = {}
                gift_prods = ["GIFT_BASKET", "ROSES", "CHOCOLATE", "STRAWBERRIES"]
                for prod in gift_prods:
                    order_depth = state.order_depths[prod]
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                    mids[prod] = (best_ask + best_bid) / 2
                component_basket_price = (
                    mids["CHOCOLATE"] * 4 + mids["STRAWBERRIES"] * 6 + mids["ROSES"]
                )
                order_depth = state.order_depths[product]

                ## GIFT BASKETS
                ## TODO: FIND OPT VALUES FOR BASKETS
                result["GIFT_BASKET"] = self.compute_gb_orders(
                    "GIFT_BASKET", state, mids, component_basket_price
                )

                result["ROSES"] = self.compute_copy_trader(state, "ROSES", "Rhianna")

                result["CHOCOLATE"] = self.compute_copy_trader(
                    state, "CHOCOLATE", "Vladimir"
                )

            if product == "COCONUT":
                mids = {}
                for prod in ["COCONUT", "COCONUT_COUPON"]:
                    order_depth = state.order_depths[product]
                    best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                    best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]

                    mids[prod] = (best_ask + best_bid) / 2

                fair_coup = round(self.black_scholes_price(mids[product]))

                result["COCONUT_COUPON"] = self.compute_orders(
                    "COCONUT_COUPON",
                    state.order_depths["COCONUT_COUPON"],
                    fair_coup - 1,
                    fair_coup + 1,
                )

        logger.flush(state, result, conversions, serialized_trader_data_DICT)
        return result, conversions, serialized_trader_data_DICT
