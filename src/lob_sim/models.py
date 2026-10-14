from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"
    MARKET = "market"


@dataclass
class Order:
    order_id: int
    participant_id: str
    side: Side
    quantity: int
    timestamp: int
    order_type: OrderType = OrderType.LIMIT
    price: int | None = None
    remaining: int = field(init=False)

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(
                "quantity must be positive"
            )

        if self.order_type == OrderType.LIMIT:
            if self.price is None:
                raise ValueError(
                    "limit orders require a price"
                )

            if self.price <= 0:
                raise ValueError(
                    "price must be positive"
                )

        if (
            self.order_type == OrderType.MARKET
            and self.price is not None
        ):
            raise ValueError(
                "market orders cannot have a price"
            )

        self.remaining = self.quantity


@dataclass(frozen=True)
class Trade:
    trade_id: int
    timestamp: int
    price: int
    quantity: int
    buyer_id: str
    seller_id: str
    maker_order_id: int
    taker_order_id: int
    aggressor_side: Side
