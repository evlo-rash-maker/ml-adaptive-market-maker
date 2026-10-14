from dataclasses import dataclass

import numpy as np
import pandas as pd

from lob_sim.market import FairValueProcess
from lob_sim.market_maker import MarketMaker
from lob_sim.models import Order, Side
from lob_sim.order_book import OrderBook
from lob_sim.traders import (
    BackgroundLiquidity,
    InformedTrader,
    NoiseTrader,
)


@dataclass(frozen=True)
class SimulationConfig:
    steps: int = 1_000
    seed: int = 42

    initial_fair_value: float = 10_000.0
    base_volatility: float = 2.0

    informed_trader_share: float = 0.15
    informed_regime_multiplier: float = 3.0
    information_shock_probability: float = 0.08
    information_shock_scale: float = 8.0

    information_regime_min_steps: int = 4
    information_regime_max_steps: int = 8

    high_volatility_probability: float = 0.15
    high_volatility_multiplier: float = 3.0

    background_levels: int = 5
    background_quantity: int = 12
    background_spacing: int = 5


@dataclass
class SimulationResult:
    snapshots: pd.DataFrame
    trades: pd.DataFrame


class MarketSimulation:
    def __init__(
        self,
        config=None,
        market_maker=None,
    ):
        self.config = (
            config
            if config is not None
            else SimulationConfig()
        )

        self.rng = np.random.default_rng(
            self.config.seed
        )

        self.book = OrderBook()

        self.fair_value_process = FairValueProcess(
            initial_value=self.config.initial_fair_value,
            base_volatility=self.config.base_volatility,
            seed=self.config.seed + 1,
        )

        self.market_maker = (
            market_maker
            if market_maker is not None
            else MarketMaker()
        )

        self.noise_trader = NoiseTrader(
            participant_id="noise",
            seed=self.config.seed + 2,
        )

        self.informed_trader = InformedTrader(
            participant_id="informed",
            signal_threshold=1.0,
            seed=self.config.seed + 3,
        )

        self.background = BackgroundLiquidity(
            participant_id="background",
            levels=self.config.background_levels,
            quantity_per_level=self.config.background_quantity,
            spacing=self.config.background_spacing,
        )

        self.information_signal = 0.0
        self.information_steps_remaining = 0

        self.next_order_id = 1

        self.market_maker_order_ids = []
        self.background_order_ids = []

        self.snapshot_rows = []
        self.trade_rows = []

    def _new_order_id(self):
        order_id = self.next_order_id
        self.next_order_id += 1
        return order_id

    def _cancel_orders(
        self,
        order_ids,
    ):
        for order_id in order_ids:
            if order_id in self.book.orders:
                self.book.cancel(order_id)

    def _submit(
        self,
        order: Order,
    ):
        trades = self.book.submit(order)

        for trade in trades:
            self.market_maker.apply_trade(
                trade
            )

            self.trade_rows.append(
                {
                    "trade_id": trade.trade_id,
                    "timestamp": trade.timestamp,
                    "price": trade.price,
                    "quantity": trade.quantity,
                    "buyer_id": trade.buyer_id,
                    "seller_id": trade.seller_id,
                    "maker_order_id": (
                        trade.maker_order_id
                    ),
                    "taker_order_id": (
                        trade.taker_order_id
                    ),
                    "aggressor_side": (
                        trade.aggressor_side.value
                    ),
                }
            )

        return trades

    def _refresh_background(
        self,
        timestamp,
        fair_value,
    ):
        self._cancel_orders(
            self.background_order_ids
        )

        self.background_order_ids = []

        orders = self.background.create_orders(
            first_order_id=self.next_order_id,
            timestamp=timestamp,
            fair_value=fair_value,
        )

        self.next_order_id += len(orders)

        for order in orders:
            self._submit(order)

            if order.remaining > 0:
                self.background_order_ids.append(
                    order.order_id
                )

    def _refresh_market_maker(
        self,
        timestamp,
        fair_value,
        volatility,
    ):
        self._cancel_orders(
            self.market_maker_order_ids
        )

        self.market_maker_order_ids = []

        orders = self.market_maker.create_orders(
            first_order_id=self.next_order_id,
            timestamp=timestamp,
            fair_value=fair_value,
            volatility=volatility,
        )

        quoted_quantity = sum(
            order.quantity
            for order in orders
        )

        self.next_order_id += len(orders)

        for order in orders:
            self._submit(order)

            if order.remaining > 0:
                self.market_maker_order_ids.append(
                    order.order_id
                )

        return quoted_quantity

    def _draw_volatility_multiplier(
        self,
    ):
        if (
            self.rng.random()
            < self.config.high_volatility_probability
        ):
            return (
                self.config.high_volatility_multiplier
            )

        return 1.0

    def _draw_information_shock(
        self,
    ):
        if self.information_steps_remaining <= 0:
            if (
                self.rng.random()
                >= self.config.information_shock_probability
            ):
                return 0.0

            duration = int(
                self.rng.integers(
                    self.config.information_regime_min_steps,
                    self.config.information_regime_max_steps + 1,
                )
            )

            direction = (
                1.0
                if self.rng.random() < 0.5
                else -1.0
            )

            magnitude = max(
                1.25,
                abs(
                    self.rng.normal(
                        loc=(
                            self.config
                            .information_shock_scale
                            / 3
                        ),
                        scale=(
                            self.config
                            .information_shock_scale
                            / 6
                        ),
                    )
                ),
            )

            self.information_signal = (
                direction * magnitude
            )

            self.information_steps_remaining = (
                duration
            )

        shock = self.information_signal

        self.information_steps_remaining -= 1

        if self.information_steps_remaining <= 0:
            self.information_signal = 0.0

        return float(shock)

    def _submit_arriving_trader(
        self,
        timestamp,
        fair_value,
        information_signal,
    ):
        signal_active = (
            abs(information_signal)
            >= self.informed_trader.signal_threshold
        )

        informed_probability = min(
            1.0,
            self.config.informed_trader_share
            * self.config.informed_regime_multiplier,
        )

        use_informed = (
            signal_active
            and self.rng.random()
            < informed_probability
        )

        if use_informed:
            order = self.informed_trader.create_order(
                order_id=self._new_order_id(),
                timestamp=timestamp,
                information_signal=information_signal,
            )

            if order is not None:
                self._submit(order)

            return "informed"

        order = self.noise_trader.create_order(
            order_id=self._new_order_id(),
            timestamp=timestamp,
            fair_value=fair_value,
            best_bid=self.book.best_bid,
            best_ask=self.book.best_ask,
        )

        self._submit(order)

        return "noise"

    def _book_depth(
        self,
        levels=3,
    ):
        bid_prices = list(
            self.book.bids.keys()
        )[-levels:]

        ask_prices = list(
            self.book.asks.keys()
        )[:levels]

        bid_depth = sum(
            self.book.quantity_at_price(
                Side.BUY,
                price,
            )
            for price in bid_prices
        )

        ask_depth = sum(
            self.book.quantity_at_price(
                Side.SELL,
                price,
            )
            for price in ask_prices
        )

        total_depth = (
            bid_depth
            + ask_depth
        )

        if total_depth == 0:
            imbalance = 0.0
        else:
            imbalance = (
                bid_depth
                - ask_depth
            ) / total_depth

        return (
            bid_depth,
            ask_depth,
            imbalance,
        )

    def step(
        self,
        timestamp,
    ):
        fair_value_before = (
            self.fair_value_process.value
        )

        volatility_multiplier = (
            self._draw_volatility_multiplier()
        )

        volatility = (
            self.config.base_volatility
            * volatility_multiplier
        )

        information_shock = (
            self._draw_information_shock()
        )

        # Remove stale MM quotes before new
        # background liquidity is submitted.
        self._cancel_orders(
            self.market_maker_order_ids
        )

        self.market_maker_order_ids = []

        self._refresh_background(
            timestamp=timestamp,
            fair_value=fair_value_before,
        )

        mm_quote_quantity = (
            self._refresh_market_maker(
                timestamp=timestamp,
                fair_value=fair_value_before,
                volatility=volatility,
            )
        )

        arriving_trader = (
            self._submit_arriving_trader(
                timestamp=timestamp,
                fair_value=fair_value_before,
                information_signal=information_shock,
            )
        )

        (
            bid_depth_3,
            ask_depth_3,
            book_imbalance_3,
        ) = self._book_depth(
            levels=3
        )

        best_bid_depth = (
            self.book.quantity_at_price(
                Side.BUY,
                self.book.best_bid,
            )
            if self.book.best_bid is not None
            else 0
        )

        best_ask_depth = (
            self.book.quantity_at_price(
                Side.SELL,
                self.book.best_ask,
            )
            if self.book.best_ask is not None
            else 0
        )

        state = self.fair_value_process.advance(
            timestamp=timestamp,
            volatility_multiplier=volatility_multiplier,
            information_shock=information_shock,
        )

        self.snapshot_rows.append(
            {
                "timestamp": timestamp,
                "fair_value_before": (
                    fair_value_before
                ),
                "fair_value_after": (
                    state.fair_value
                ),
                "volatility": volatility,
                "information_shock": (
                    information_shock
                ),
                "arriving_trader": (
                    arriving_trader
                ),
                "inventory": (
                    self.market_maker.inventory
                ),
                "best_bid": (
                    self.book.best_bid
                ),
                "best_ask": (
                    self.book.best_ask
                ),
                "spread": (
                    self.book.spread
                ),
                "best_bid_depth": (
                    best_bid_depth
                ),
                "best_ask_depth": (
                    best_ask_depth
                ),
                "bid_depth_3": (
                    bid_depth_3
                ),
                "ask_depth_3": (
                    ask_depth_3
                ),
                "book_imbalance_3": (
                    book_imbalance_3
                ),
                "mm_quote_quantity": (
                    mm_quote_quantity
                ),
            }
        )

    def run(self):
        for timestamp in range(
            self.config.steps
        ):
            self.step(timestamp)

        snapshots = pd.DataFrame(
            self.snapshot_rows
        )

        trades = pd.DataFrame(
            self.trade_rows
        )

        return SimulationResult(
            snapshots=snapshots,
            trades=trades,
        )