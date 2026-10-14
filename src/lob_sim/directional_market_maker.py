import math

from lob_sim.market_maker import MarketMaker
from lob_sim.models import (
    Order,
    OrderType,
    Side,
)


class DirectionalMLMarketMaker(
    MarketMaker
):
    def __init__(
        self,
        participant_id="market_maker",
        base_half_spread=2.0,
        volatility_sensitivity=0.5,
        inventory_skew=0.25,
        max_inventory=25,
        quote_size=5,
        risk_threshold=0.45,
        risk_spread_sensitivity=2.0,
        direction_sensitivity=4.0,
        size_sensitivity=0.25,
    ):
        super().__init__(
            participant_id=participant_id,
            base_half_spread=base_half_spread,
            volatility_sensitivity=volatility_sensitivity,
            inventory_skew=inventory_skew,
            max_inventory=max_inventory,
            quote_size=quote_size,
        )

        self.risk_threshold = float(
            risk_threshold
        )

        self.risk_spread_sensitivity = float(
            risk_spread_sensitivity
        )

        self.direction_sensitivity = float(
            direction_sensitivity
        )

        self.size_sensitivity = float(
            size_sensitivity
        )

        self.normal_probability = 1.0
        self.buy_probability = 0.0
        self.sell_probability = 0.0

    def update_flow_probabilities(
        self,
        normal,
        buy,
        sell,
    ):
        self.normal_probability = float(
            normal
        )

        self.buy_probability = float(
            buy
        )

        self.sell_probability = float(
            sell
        )

    def create_orders(
        self,
        first_order_id,
        timestamp,
        fair_value,
        volatility,
    ):
        informed_risk = (
            self.buy_probability
            + self.sell_probability
        )

        directional_signal = (
            self.buy_probability
            - self.sell_probability
        )

        reservation_price = (
            fair_value
            - self.inventory_skew
            * self.inventory
            + self.direction_sensitivity
            * directional_signal
        )

        risk_excess = max(
            0.0,
            informed_risk
            - self.risk_threshold,
        )

        half_spread = (
            self.base_half_spread
            + self.volatility_sensitivity
            * volatility
            + self.risk_spread_sensitivity
            * risk_excess
        )

        bid_price = max(
            1,
            math.floor(
                reservation_price
                - half_spread
            ),
        )

        ask_price = max(
            bid_price + 1,
            math.ceil(
                reservation_price
                + half_spread
            ),
        )

        size_multiplier = max(
            0.50,
            1.0
            - self.size_sensitivity
            * informed_risk,
        )

        size = max(
            1,
            int(
                round(
                    self.quote_size
                    * size_multiplier
                )
            ),
        )

        orders = []
        order_id = first_order_id

        if (
            self.inventory + size
            <= self.max_inventory
        ):
            orders.append(
                Order(
                    order_id=order_id,
                    participant_id=self.participant_id,
                    side=Side.BUY,
                    quantity=size,
                    timestamp=timestamp,
                    order_type=OrderType.LIMIT,
                    price=bid_price,
                )
            )

            order_id += 1

        if (
            self.inventory - size
            >= -self.max_inventory
        ):
            orders.append(
                Order(
                    order_id=order_id,
                    participant_id=self.participant_id,
                    side=Side.SELL,
                    quantity=size,
                    timestamp=timestamp,
                    order_type=OrderType.LIMIT,
                    price=ask_price,
                )
            )

        return orders
