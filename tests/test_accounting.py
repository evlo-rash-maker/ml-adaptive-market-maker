import numpy as np

from lob_sim.accounting import (
    analyze_market_maker,
)
from lob_sim.market_maker import (
    MarketMaker,
)
from lob_sim.simulation import (
    MarketSimulation,
    SimulationConfig,
)


def make_result(seed=42):
    config = SimulationConfig(
        steps=500,
        seed=seed,
    )

    maker = MarketMaker(
        max_inventory=25,
        quote_size=5,
    )

    return MarketSimulation(
        config=config,
        market_maker=maker,
    ).run()


def test_accounting_produces_timeline():
    result = make_result()

    accounting = (
        analyze_market_maker(
            result
        )
    )

    assert len(
        accounting.timeline
    ) == len(
        result.snapshots
    )


def test_only_market_maker_fills_are_kept():
    accounting = (
        analyze_market_maker(
            make_result()
        )
    )

    fills = accounting.fills

    assert (
        fills["buyer_id"].eq(
            "market_maker"
        )
        | fills["seller_id"].eq(
            "market_maker"
        )
    ).all()


def test_pnl_decomposition_adds_up():
    accounting = (
        analyze_market_maker(
            make_result()
        )
    )

    summary = accounting.summary

    decomposed = (
        summary.spread_capture
        + summary.selection_pnl
        + summary.inventory_carry_pnl
    )

    assert np.isclose(
        summary.total_pnl,
        decomposed,
    )


def test_accounting_matches_mark_to_market():
    accounting = (
        analyze_market_maker(
            make_result()
        )
    )

    assert np.isclose(
        accounting.summary.total_pnl,
        accounting.summary.final_equity,
    )


def test_adverse_selection_cost_is_nonnegative():
    accounting = (
        analyze_market_maker(
            make_result()
        )
    )

    assert (
        accounting.summary
        .adverse_selection_cost
        >= 0
    )


def test_inventory_matches_simulation():
    result = make_result()

    accounting = (
        analyze_market_maker(
            result
        )
    )

    assert np.array_equal(
        accounting.timeline[
            "inventory"
        ].to_numpy(),
        result.snapshots[
            "inventory"
        ].to_numpy(),
    )