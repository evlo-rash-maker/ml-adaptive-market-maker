import pandas as pd

from lob_sim.market_maker import (
    MarketMaker,
)
from lob_sim.simulation import (
    MarketSimulation,
    SimulationConfig,
)


def test_simulation_runs_requested_steps():
    config = SimulationConfig(
        steps=100,
        seed=10,
    )

    simulation = MarketSimulation(
        config=config
    )

    result = simulation.run()

    assert len(
        result.snapshots
    ) == 100


def test_simulation_produces_trades():
    config = SimulationConfig(
        steps=300,
        seed=20,
    )

    simulation = MarketSimulation(
        config=config
    )

    result = simulation.run()

    assert not result.trades.empty


def test_simulation_is_reproducible():
    config = SimulationConfig(
        steps=100,
        seed=123,
    )

    first = MarketSimulation(
        config=config
    ).run()

    second = MarketSimulation(
        config=config
    ).run()

    pd.testing.assert_frame_equal(
        first.snapshots,
        second.snapshots,
    )

    pd.testing.assert_frame_equal(
        first.trades,
        second.trades,
    )


def test_inventory_respects_limit():
    config = SimulationConfig(
        steps=500,
        seed=99,
    )

    maker = MarketMaker(
        max_inventory=20,
        quote_size=5,
    )

    result = MarketSimulation(
        config=config,
        market_maker=maker,
    ).run()

    assert (
        result.snapshots[
            "inventory"
        ].abs().max()
        <= 20
    )


def test_snapshot_contains_market_state():
    config = SimulationConfig(
        steps=20,
        seed=4,
    )

    result = MarketSimulation(
        config=config
    ).run()

    expected = {
        "timestamp",
        "fair_value_before",
        "fair_value_after",
        "volatility",
        "information_shock",
        "arriving_trader",
        "inventory",
        "best_bid",
        "best_ask",
        "spread",
    }

    assert expected.issubset(
        result.snapshots.columns
    )