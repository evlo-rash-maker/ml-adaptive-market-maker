from lob_sim.market import (
    FairValueProcess,
)
from lob_sim.models import (
    OrderType,
    Side,
)
from lob_sim.traders import (
    BackgroundLiquidity,
    InformedTrader,
)


def test_fair_value_is_reproducible():
    first = FairValueProcess(
        seed=123
    )

    second = FairValueProcess(
        seed=123
    )

    first_values = [
        first.advance(t).fair_value
        for t in range(20)
    ]

    second_values = [
        second.advance(t).fair_value
        for t in range(20)
    ]

    assert first_values == second_values


def test_information_shock_moves_value():
    process = FairValueProcess(
        initial_value=100,
        base_volatility=0,
        mean_reversion=0,
    )

    state = process.advance(
        timestamp=1,
        information_shock=10,
    )

    assert state.fair_value == 110


def test_informed_trader_buys_positive_signal():
    trader = InformedTrader(
        participant_id="informed",
        signal_threshold=1,
    )

    order = trader.create_order(
        order_id=1,
        timestamp=1,
        information_signal=5,
    )

    assert order.side == Side.BUY
    assert order.order_type == OrderType.MARKET


def test_informed_trader_sells_negative_signal():
    trader = InformedTrader(
        participant_id="informed",
        signal_threshold=1,
    )

    order = trader.create_order(
        order_id=1,
        timestamp=1,
        information_signal=-5,
    )

    assert order.side == Side.SELL


def test_small_signal_creates_no_trade():
    trader = InformedTrader(
        participant_id="informed",
        signal_threshold=2,
    )

    order = trader.create_order(
        order_id=1,
        timestamp=1,
        information_signal=0.5,
    )

    assert order is None


def test_background_liquidity_builds_both_sides():
    provider = BackgroundLiquidity(
        levels=3,
        quantity_per_level=10,
        spacing=2,
    )

    orders = provider.create_orders(
        first_order_id=1,
        timestamp=1,
        fair_value=100,
    )

    buys = [
        order
        for order in orders
        if order.side == Side.BUY
    ]

    sells = [
        order
        for order in orders
        if order.side == Side.SELL
    ]

    assert len(buys) == 3
    assert len(sells) == 3

    assert max(
        order.price
        for order in buys
    ) < 100

    assert min(
        order.price
        for order in sells
    ) > 100