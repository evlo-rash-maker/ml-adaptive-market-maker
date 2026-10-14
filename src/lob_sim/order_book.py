from collections import deque

from sortedcontainers import SortedDict

from lob_sim.models import Order, OrderType, Side, Trade


class OrderBook:
    def __init__(self):
        self.bids = SortedDict()
        self.asks = SortedDict()

        self.orders = {}
        self.seen_order_ids = set()

        self._next_trade_id = 1

    @property
    def best_bid(self):
        if not self.bids:
            return None

        return self.bids.peekitem(-1)[0]

    @property
    def best_ask(self):
        if not self.asks:
            return None

        return self.asks.peekitem(0)[0]

    @property
    def spread(self):
        if self.best_bid is None or self.best_ask is None:
            return None

        return self.best_ask - self.best_bid

    @property
    def mid_price(self):
        if self.best_bid is None or self.best_ask is None:
            return None

        return (self.best_bid + self.best_ask) / 2

    def submit(self, order: Order) -> list[Trade]:
        if order.order_id in self.seen_order_ids:
            raise ValueError(
                f"duplicate order id: {order.order_id}"
            )

        self.seen_order_ids.add(order.order_id)

        trades = []

        while order.remaining > 0 and self._can_match(order):
            trades.append(
                self._match_once(order)
            )

        if (
            order.remaining > 0
            and order.order_type == OrderType.LIMIT
        ):
            self._rest(order)

        return trades

    def cancel(self, order_id: int) -> Order:
        if order_id not in self.orders:
            raise KeyError(
                f"unknown active order: {order_id}"
            )

        order = self.orders.pop(order_id)

        book = self._book_for_side(
            order.side
        )

        queue = book[order.price]

        queue.remove(order_id)

        if not queue:
            del book[order.price]

        return order

    def quantity_at_price(
        self,
        side: Side,
        price: int,
    ) -> int:
        book = self._book_for_side(side)

        if price not in book:
            return 0

        return sum(
            self.orders[order_id].remaining
            for order_id in book[price]
        )

    def _rest(self, order: Order):
        book = self._book_for_side(
            order.side
        )

        if order.price not in book:
            book[order.price] = deque()

        book[order.price].append(
            order.order_id
        )

        self.orders[
            order.order_id
        ] = order

    def _can_match(
        self,
        incoming: Order,
    ) -> bool:
        if incoming.side == Side.BUY:
            if not self.asks:
                return False

            if incoming.order_type == OrderType.MARKET:
                return True

            return self.best_ask <= incoming.price

        if not self.bids:
            return False

        if incoming.order_type == OrderType.MARKET:
            return True

        return self.best_bid >= incoming.price

    def _match_once(
        self,
        incoming: Order,
    ) -> Trade:
        if incoming.side == Side.BUY:
            book = self.asks
            price = self.best_ask

        else:
            book = self.bids
            price = self.best_bid

        queue = book[price]

        resting_id = queue[0]

        resting = self.orders[
            resting_id
        ]

        quantity = min(
            incoming.remaining,
            resting.remaining,
        )

        incoming.remaining -= quantity
        resting.remaining -= quantity

        if incoming.side == Side.BUY:
            buyer_id = incoming.participant_id
            seller_id = resting.participant_id

        else:
            buyer_id = resting.participant_id
            seller_id = incoming.participant_id

        trade = Trade(
            trade_id=self._next_trade_id,
            timestamp=incoming.timestamp,
            price=price,
            quantity=quantity,
            buyer_id=buyer_id,
            seller_id=seller_id,
            maker_order_id=resting.order_id,
            taker_order_id=incoming.order_id,
            aggressor_side=incoming.side,
        )

        self._next_trade_id += 1

        if resting.remaining == 0:
            queue.popleft()

            del self.orders[
                resting.order_id
            ]

            if not queue:
                del book[price]

        return trade

    def _book_for_side(
        self,
        side: Side,
    ):
        if side == Side.BUY:
            return self.bids

        return self.asks

   