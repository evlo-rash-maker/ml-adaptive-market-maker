import json

import pandas as pd

from lob_sim.accounting import analyze_market_maker
from lob_sim.directional_market_maker import (
    DirectionalMLMarketMaker,
)
from lob_sim.directional_signal import (
    DirectionalFlowPredictor,
)
from lob_sim.simulation import (
    MarketSimulation,
    SimulationConfig,
    SimulationResult,
)


SEEDS = range(50, 53)
STEPS = 300

THRESHOLDS = [
    0.45,
    0.55,
]

DIRECTION_SENSITIVITIES = [
    2.0,
    4.0,
]

RISK_SPREAD_SENSITIVITIES = [
    0.0,
    1.0,
]


def run(
    seed,
    threshold,
    direction_sensitivity,
    risk_spread_sensitivity,
):
    config = SimulationConfig(
        steps=STEPS,
        seed=seed,
    )

    maker = DirectionalMLMarketMaker(
        risk_threshold=threshold,
        direction_sensitivity=(
            direction_sensitivity
        ),
        risk_spread_sensitivity=(
            risk_spread_sensitivity
        ),
    )

    simulation = MarketSimulation(
        config=config,
        market_maker=maker,
    )

    predictor = DirectionalFlowPredictor()

    cursor = 0

    for timestamp in range(
        config.steps
    ):
        simulation.step(timestamp)

        trades = simulation.trade_rows[
            cursor:
        ]

        cursor = len(
            simulation.trade_rows
        )

        predictor.observe(
            simulation.snapshot_rows[-1],
            trades,
        )

        probabilities = (
            predictor.predict_next()
        )

        maker.update_flow_probabilities(
            **probabilities
        )

    result = SimulationResult(
        snapshots=pd.DataFrame(
            simulation.snapshot_rows
        ),
        trades=pd.DataFrame(
            simulation.trade_rows
        ),
    )

    return analyze_market_maker(
        result
    ).summary


rows = []

for threshold in THRESHOLDS:
    for direction in DIRECTION_SENSITIVITIES:
        for spread in RISK_SPREAD_SENSITIVITIES:

            pnl = []
            adverse = []

            for seed in SEEDS:
                summary = run(
                    seed,
                    threshold,
                    direction,
                    spread,
                )

                pnl.append(
                    summary.total_pnl
                )

                adverse.append(
                    summary.adverse_selection_cost
                )

            row = {
                "risk_threshold":
                    threshold,
                "direction_sensitivity":
                    direction,
                "risk_spread_sensitivity":
                    spread,
                "mean_pnl":
                    sum(pnl) / len(pnl),
                "mean_adverse":
                    sum(adverse)
                    / len(adverse),
            }

            rows.append(row)

            print(
                f"threshold={threshold} "
                f"direction={direction} "
                f"spread={spread} "
                f"pnl={row['mean_pnl']:.2f}"
            )


results = pd.DataFrame(rows)

best = (
    results
    .sort_values(
        "mean_pnl",
        ascending=False,
    )
    .iloc[0]
)

params = {
    "risk_threshold":
        float(
            best["risk_threshold"]
        ),
    "direction_sensitivity":
        float(
            best[
                "direction_sensitivity"
            ]
        ),
    "risk_spread_sensitivity":
        float(
            best[
                "risk_spread_sensitivity"
            ]
        ),
    "size_sensitivity":
        0.25,
}

results.to_csv(
    "results/tables/"
    "directional_ai_tuning.csv",
    index=False,
)

with open(
    "results/models/"
    "directional_strategy.json",
    "w",
) as file:
    json.dump(
        params,
        file,
        indent=2,
    )

print()
print("BEST VALIDATION STRATEGY")
print("========================")
print(params)
print(
    f"Mean validation P&L: "
    f"{best['mean_pnl']:.2f}"
)
