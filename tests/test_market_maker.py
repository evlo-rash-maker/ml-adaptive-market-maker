from lob_sim.market_maker import (
    MarketMaker,
)
from lob_sim.models import (
    Side,
    Trade,
)


def test_market_maker_quotes_both_sides():
    maker = MarketMaker(
        base_half_spread=2,
        volatility_sensitivity=0,
        inventory_skew=0,
    )

    quote = maker.calculate_quote(
        fair_value=100,
        volatility=1,
    )

    assert quote.bid_price == 98
    assert quote.ask_price == 102


def test_volatility_widens_spread():
    maker = MarketMaker(
        base_half_spread=2,
        volatility_sensitivity=2,
        inventory_skew=0,
    )

    calm = maker.calculate_quote(
        fair_value=100,
        volatility=1,
    )

    volatile = maker.calculate_quote(
        fair_value=100,
        volatility=5,
    )

    calm_width = (
        calm.ask_price
        - calm.bid_price
    )

    volatile_width = (
        volatile.ask_price
        - volatile.bid_price
    )

    assert volatile_width > calm_width


def test_long_inventory_skews_quotes_lower():
    maker = MarketMaker(
        base_half_spread=2,
        volatility_sensitivity=0,
        inventory_skew=0.5,
    )

    neutral = maker.calculate_quote(
        fair_value=100,
        volatility=0,
    )

    maker.inventory = 10

    long_inventory = maker.calculate_quote(
        fair_value=100,
        volatility=0,
    )

    assert (
        long_inventory.bid_price
        < neutral.bid_price
    )

    assert (
        long_inventory.ask_price
        < neutral.ask_price
    )


def test_inventory_limit_disables_buy_quote():
    maker = MarketMaker(
        max_inventory=10,
        quote_size=5,
    )

    maker.inventory = 8

    quote = maker.calculate_quote(
        fair_value=100,
        volatility=1,
    )

    assert quote.bid_price is None
    assert quote.ask_price is not None


def test_buy_fill_increases_inventory():
    maker = MarketMaker()

    trade = Trade(
        trade_id=1,
        timestamp=1,
        price=100,
        quantity=4,
        buyer_id="market_maker",
        seller_id="noise",
        maker_order_id=1,
        taker_order_id=2,
        aggressor_side=Side.SELL,
    )

    maker.apply_trade(trade)

    assert maker.inventory == 4


def test_sell_fill_decreases_inventory():
    maker = MarketMaker()

    trade = Trade(
        trade_id=1,
        timestamp=1,
        price=100,
        quantity=3,
        buyer_id="noise",
        seller_id="market_maker",
        maker_order_id=1,
        taker_order_id=2,
        aggressor_side=Side.BUY,
    )

    maker.apply_trade(trade)

    assert maker.inventory == -3