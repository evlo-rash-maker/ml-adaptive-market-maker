import math

import numpy as np

from lob_sim.models import (
    Order,
    OrderType,
    Side,
)


class NoiseTrader:
    def __init__(
        self,
        participant_id,
        market_order_probability=0.7,
        max_quantity=5,
        max_limit_offset=4,
        seed=1,
    ):
        self.participant_id = participant_id

        self.market_order_probability = float(
            market_order_probability
        )

        self.max_quantity = int(
            max_quantity
        )

        self.max_limit_offset = int(
            max_limit_offset
        )

        self.rng = np.random.default_rng(
            seed
        )

    def create_order(
        self,
        order_id,
        timestamp,
        fair_value,
        best_bid=None,
        best_ask=None,
    ):
        side = (
            Side.BUY
            if self.rng.random() < 0.5
            else Side.SELL
        )

        quantity = int(
            self.rng.integers(
                1,
                self.max_quantity + 1,
            )
        )

        if (
            self.rng.random()
            < self.market_order_probability
        ):
            return Order(
                order_id=order_id,
                participant_id=self.participant_id,
                side=side,
                quantity=quantity,
                timestamp=timestamp,
                order_type=OrderType.MARKET,
            )

        offset = int(
            self.rng.integers(
                1,
                self.max_limit_offset + 1,
            )
        )

        if side == Side.BUY:
            price = (
                math.floor(fair_value)
                - offset
            )

            if best_ask is not None:
                price = min(
                    price,
                    best_ask - 1,
                )

        else:
            price = (
                math.ceil(fair_value)
                + offset
            )

            if best_bid is not None:
                price = max(
                    price,
                    best_bid + 1,
                )

        price = max(
            1,
            int(price),
        )

        return Order(
            order_id=order_id,
            participant_id=self.participant_id,
            side=side,
            quantity=quantity,
            timestamp=timestamp,
            order_type=OrderType.LIMIT,
            price=price,
        )


class InformedTrader:
    def __init__(
        self,
        participant_id,
        signal_threshold=1.0,
        max_quantity=8,
        seed=2,
    ):
        self.participant_id = participant_id

        self.signal_threshold = float(
            signal_threshold
        )

        self.max_quantity = int(
            max_quantity
        )

        self.rng = np.random.default_rng(
            seed
        )

    def create_order(
        self,
        order_id,
        timestamp,
        information_signal,
    ):
        if (
            abs(information_signal)
            < self.signal_threshold
        ):
            return None

        if information_signal > 0:
            side = Side.BUY
        else:
            side = Side.SELL

        quantity = int(
            self.rng.integers(
                1,
                self.max_quantity + 1,
            )
        )

        return Order(
            order_id=order_id,
            participant_id=self.participant_id,
            side=side,
            quantity=quantity,
            timestamp=timestamp,
            order_type=OrderType.MARKET,
        )


class BackgroundLiquidity:
    def __init__(
        self,
        participant_id="background",
        levels=5,
        quantity_per_level=10,
        spacing=2,
    ):
        self.participant_id = participant_id
        self.levels = int(levels)

        self.quantity_per_level = int(
            quantity_per_level
        )

        self.spacing = int(spacing)

    def create_orders(
        self,
        first_order_id,
        timestamp,
        fair_value,
    ):
        orders = []

        center = int(
            round(fair_value)
        )

        order_id = first_order_id

        for level in range(
            1,
            self.levels + 1,
        ):
            distance = (
                level
                * self.spacing
            )

            orders.append(
                Order(
                    order_id=order_id,
                    participant_id=self.participant_id,
                    side=Side.BUY,
                    quantity=self.quantity_per_level,
                    timestamp=timestamp,
                    order_type=OrderType.LIMIT,
                    price=max(
                        1,
                        center - distance,
                    ),
                )
            )

            order_id += 1

            orders.append(
                Order(
                    order_id=order_id,
                    participant_id=self.participant_id,
                    side=Side.SELL,
                    quantity=self.quantity_per_level,
                    timestamp=timestamp,
                    order_type=OrderType.LIMIT,
                    price=(
                        center + distance
                    ),
                )
            )

            order_id += 1

        return orders