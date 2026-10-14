from lob_sim.models import Order, OrderType, Side
from lob_sim.order_book import OrderBook


def limit_order(
    order_id,
    participant,
    side,
    price,
    quantity,
    timestamp,
):
    return Order(
        order_id=order_id,
        participant_id=participant,
        side=side,
        price=price,
        quantity=quantity,
        timestamp=timestamp,
        order_type=OrderType.LIMIT,
    )


def market_order(
    order_id,
    participant,
    side,
    quantity,
    timestamp,
):
    return Order(
        order_id=order_id,
        participant_id=participant,
        side=side,
        quantity=quantity,
        timestamp=timestamp,
        order_type=OrderType.MARKET,
    )


def test_best_bid_and_ask():
    book = OrderBook()

    book.submit(
        limit_order(
            1,
            "buyer",
            Side.BUY,
            99,
            5,
            1,
        )
    )

    book.submit(
        limit_order(
            2,
            "seller",
            Side.SELL,
            101,
            5,
            2,
        )
    )

    assert book.best_bid == 99
    assert book.best_ask == 101
    assert book.spread == 2
    assert book.mid_price == 100


def test_price_time_priority():
    book = OrderBook()

    book.submit(
        limit_order(
            1,
            "seller_1",
            Side.SELL,
            101,
            5,
            1,
        )
    )

    book.submit(
        limit_order(
            2,
            "seller_2",
            Side.SELL,
            101,
            5,
            2,
        )
    )

    trades = book.submit(
        market_order(
            3,
            "buyer",
            Side.BUY,
            7,
            3,
        )
    )

    assert len(trades) == 2
    assert trades[0].seller_id == "seller_1"
    assert trades[0].quantity == 5
    assert trades[1].seller_id == "seller_2"
    assert trades[1].quantity == 2


def test_market_order_walks_book():
    book = OrderBook()

    book.submit(
        limit_order(
            1,
            "seller_1",
            Side.SELL,
            101,
            2,
            1,
        )
    )

    book.submit(
        limit_order(
            2,
            "seller_2",
            Side.SELL,
            102,
            3,
            2,
        )
    )

    trades = book.submit(
        market_order(
            3,
            "buyer",
            Side.BUY,
            4,
            3,
        )
    )

    assert [trade.price for trade in trades] == [101, 102]
    assert [trade.quantity for trade in trades] == [2, 2]
    assert book.best_ask == 102
    assert book.quantity_at_price(Side.SELL, 102) == 1


def test_crossing_limit_order():
    book = OrderBook()

    book.submit(
        limit_order(
            1,
            "seller",
            Side.SELL,
            101,
            5,
            1,
        )
    )

    trades = book.submit(
        limit_order(
            2,
            "buyer",
            Side.BUY,
            102,
            3,
            2,
        )
    )

    assert len(trades) == 1
    assert trades[0].price == 101
    assert trades[0].quantity == 3


def test_cancel_order():
    book = OrderBook()

    book.submit(
        limit_order(
            1,
            "buyer",
            Side.BUY,
            99,
            5,
            1,
        )
    )

    book.cancel(1)

    assert book.best_bid is None
    assert 1 not in book.orders


def test_partial_fill_remains_in_book():
    book = OrderBook()

    book.submit(
        limit_order(
            1,
            "seller",
            Side.SELL,
            101,
            10,
            1,
        )
    )

    book.submit(
        market_order(
            2,
            "buyer",
            Side.BUY,
            4,
            2,
        )
    )

    assert book.quantity_at_price(
        Side.SELL,
        101,
    ) == 6