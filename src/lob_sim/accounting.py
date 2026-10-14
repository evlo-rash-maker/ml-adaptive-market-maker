from dataclasses import dataclass

import numpy as np
import pandas as pd

from lob_sim.simulation import SimulationResult


@dataclass(frozen=True)
class PerformanceSummary:
    total_pnl: float
    final_cash: float
    final_inventory: int
    final_equity: float

    spread_capture: float
    selection_pnl: float
    adverse_selection_cost: float
    inventory_carry_pnl: float

    market_maker_trade_count: int
    filled_quantity: int
    quoted_quantity: int
    fill_rate: float
    max_absolute_inventory: int


@dataclass
class AccountingResult:
    timeline: pd.DataFrame
    fills: pd.DataFrame
    summary: PerformanceSummary


def analyze_market_maker(
    result: SimulationResult,
    participant_id="market_maker",
):
    snapshots = result.snapshots.copy()
    trades = result.trades.copy()

    snapshots = snapshots.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    if trades.empty:
        fills = pd.DataFrame()
    else:
        mask = (
            trades["buyer_id"].eq(
                participant_id
            )
            | trades["seller_id"].eq(
                participant_id
            )
        )

        fills = (
            trades.loc[mask]
            .copy()
            .reset_index(drop=True)
        )

    market_values = snapshots[
        [
            "timestamp",
            "fair_value_before",
            "fair_value_after",
        ]
    ]

    if fills.empty:
        fills = pd.DataFrame(
            columns=[
                "timestamp",
                "quantity",
                "price",
                "signed_quantity",
                "cash_change",
                "spread_capture",
                "selection_pnl",
                "adverse_selection_cost",
            ]
        )

    else:
        fills = fills.merge(
            market_values,
            on="timestamp",
            how="left",
            validate="many_to_one",
        )

        fills["signed_quantity"] = np.where(
            fills["buyer_id"].eq(
                participant_id
            ),
            fills["quantity"],
            -fills["quantity"],
        )

        fills["cash_change"] = (
            -fills["signed_quantity"]
            * fills["price"]
        )

        fills["spread_capture"] = (
            fills["signed_quantity"]
            * (
                fills["fair_value_before"]
                - fills["price"]
            )
        )

        fills["selection_pnl"] = (
            fills["signed_quantity"]
            * (
                fills["fair_value_after"]
                - fills["fair_value_before"]
            )
        )

        fills["adverse_selection_cost"] = (
            -fills["selection_pnl"]
        ).clip(lower=0)

    if fills.empty:
        per_step = pd.DataFrame(
            columns=[
                "timestamp",
                "net_trade_quantity",
                "cash_change",
                "spread_capture",
                "selection_pnl",
                "adverse_selection_cost",
                "filled_quantity",
                "trade_count",
            ]
        )

    else:
        working = fills.copy()

        working["filled_quantity"] = (
            working[
                "signed_quantity"
            ].abs()
        )

        per_step = (
            working.groupby(
                "timestamp",
                as_index=False,
            )
            .agg(
                net_trade_quantity=(
                    "signed_quantity",
                    "sum",
                ),
                cash_change=(
                    "cash_change",
                    "sum",
                ),
                spread_capture=(
                    "spread_capture",
                    "sum",
                ),
                selection_pnl=(
                    "selection_pnl",
                    "sum",
                ),
                adverse_selection_cost=(
                    "adverse_selection_cost",
                    "sum",
                ),
                filled_quantity=(
                    "filled_quantity",
                    "sum",
                ),
                trade_count=(
                    "trade_id",
                    "count",
                ),
            )
        )

    timeline = snapshots.merge(
        per_step,
        on="timestamp",
        how="left",
    )

    zero_columns = [
        "net_trade_quantity",
        "cash_change",
        "spread_capture",
        "selection_pnl",
        "adverse_selection_cost",
        "filled_quantity",
        "trade_count",
    ]

    timeline[zero_columns] = (
        timeline[zero_columns]
        .fillna(0)
    )

    timeline[
        "inventory_before_trade"
    ] = (
        timeline["inventory"]
        - timeline[
            "net_trade_quantity"
        ]
    )

    timeline["fair_value_change"] = (
        timeline["fair_value_after"]
        - timeline["fair_value_before"]
    )

    timeline[
        "inventory_carry_pnl"
    ] = (
        timeline[
            "inventory_before_trade"
        ]
        * timeline[
            "fair_value_change"
        ]
    )

    timeline["step_pnl"] = (
        timeline["spread_capture"]
        + timeline["selection_pnl"]
        + timeline[
            "inventory_carry_pnl"
        ]
    )

    timeline["cash"] = (
        timeline[
            "cash_change"
        ].cumsum()
    )

    timeline["equity"] = (
        timeline["cash"]
        + timeline["inventory"]
        * timeline["fair_value_after"]
    )

    timeline[
        "cumulative_pnl"
    ] = timeline[
        "step_pnl"
    ].cumsum()

    final_cash = float(
        timeline["cash"].iloc[-1]
    )

    final_inventory = int(
        timeline[
            "inventory"
        ].iloc[-1]
    )

    final_equity = float(
        timeline[
            "equity"
        ].iloc[-1]
    )

    total_pnl = float(
        timeline[
            "cumulative_pnl"
        ].iloc[-1]
    )

    quoted_quantity = int(
        timeline["mm_quote_quantity"].sum()
    )

    filled_quantity = int(
        timeline["filled_quantity"].sum()
    )

    fill_rate = (
        filled_quantity / quoted_quantity
        if quoted_quantity > 0
        else 0.0
    )

    summary = PerformanceSummary(
        total_pnl=total_pnl,
        final_cash=final_cash,
        final_inventory=final_inventory,
        final_equity=final_equity,
        spread_capture=float(
            timeline[
                "spread_capture"
            ].sum()
        ),
        selection_pnl=float(
            timeline[
                "selection_pnl"
            ].sum()
        ),
        adverse_selection_cost=float(
            timeline[
                "adverse_selection_cost"
            ].sum()
        ),
        inventory_carry_pnl=float(
            timeline[
                "inventory_carry_pnl"
            ].sum()
        ),
        market_maker_trade_count=int(
            timeline[
                "trade_count"
            ].sum()
        ),
        filled_quantity=filled_quantity,
        quoted_quantity=quoted_quantity,
        fill_rate=float(fill_rate),
        max_absolute_inventory=int(
            timeline[
                "inventory"
            ].abs().max()
        ),
    )

    return AccountingResult(
        timeline=timeline,
        fills=fills,
        summary=summary,
    )