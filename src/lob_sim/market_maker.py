import math
from dataclasses import dataclass

from lob_sim.models import (
    Order,
    OrderType,
    Side,
    Trade,
)


@dataclass(frozen=True)
class Quote:
    bid_price: int | None
    ask_price: int | None
    reservation_price: float
    half_spread: float


class MarketMaker:
    def __init__(
        self,
        participant_id="market_maker",
        base_half_spread=2.0,
        volatility_sensitivity=0.5,
        inventory_skew=0.25,
        max_inventory=25,
        quote_size=5,
    ):
        self.participant_id = participant_id

        self.base_half_spread = float(
            base_half_spread
        )

        self.volatility_sensitivity = float(
            volatility_sensitivity
        )

        self.inventory_skew = float(
            inventory_skew
        )

        self.max_inventory = int(
            max_inventory
        )

        self.quote_size = int(
            quote_size
        )

        self.inventory = 0

    def calculate_quote(
        self,
        fair_value,
        volatility,
    ):
        half_spread = (
            self.base_half_spread
            + self.volatility_sensitivity
            * volatility
        )

        reservation_price = (
            fair_value
            - self.inventory_skew
            * self.inventory
        )

        bid_price = math.floor(
            reservation_price
            - half_spread
        )

        ask_price = math.ceil(
            reservation_price
            + half_spread
        )

        bid_price = max(
            1,
            bid_price,
        )

        ask_price = max(
            bid_price + 1,
            ask_price,
        )

        if (
            self.inventory
            + self.quote_size
            > self.max_inventory
        ):
            bid_price = None

        if (
            self.inventory
            - self.quote_size
            < -self.max_inventory
        ):
            ask_price = None

        return Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            reservation_price=reservation_price,
            half_spread=half_spread,
        )

    def create_orders(
        self,
        first_order_id,
        timestamp,
        fair_value,
        volatility,
    ):
        quote = self.calculate_quote(
            fair_value=fair_value,
            volatility=volatility,
        )

        orders = []

        order_id = first_order_id

        if quote.bid_price is not None:
            orders.append(
                Order(
                    order_id=order_id,
                    participant_id=self.participant_id,
                    side=Side.BUY,
                    quantity=self.quote_size,
                    timestamp=timestamp,
                    order_type=OrderType.LIMIT,
                    price=quote.bid_price,
                )
            )

            order_id += 1

        if quote.ask_price is not None:
            orders.append(
                Order(
                    order_id=order_id,
                    participant_id=self.participant_id,
                    side=Side.SELL,
                    quantity=self.quote_size,
                    timestamp=timestamp,
                    order_type=OrderType.LIMIT,
                    price=quote.ask_price,
                )
            )

        return orders

    def apply_trade(
        self,
        trade: Trade,
    ):
        if trade.buyer_id == self.participant_id:
            self.inventory += trade.quantity

        if trade.seller_id == self.participant_id:
            self.inventory -= trade.quantity