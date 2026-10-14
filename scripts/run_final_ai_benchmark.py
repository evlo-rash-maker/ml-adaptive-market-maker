import json
import math

import pandas as pd

from lob_sim.accounting import analyze_market_maker
from lob_sim.directional_market_maker import DirectionalMLMarketMaker
from lob_sim.directional_signal import DirectionalFlowPredictor
from lob_sim.market_maker import MarketMaker
from lob_sim.simulation import (
    MarketSimulation,
    SimulationConfig,
    SimulationResult,
)


SEEDS = range(100, 112)
STEPS = 600


with open(
    "results/models/directional_strategy.json"
) as file:
    AI_PARAMS = json.load(file)


def run_baseline(seed):
    result = MarketSimulation(
        config=SimulationConfig(
            steps=STEPS,
            seed=seed,
        ),
        market_maker=MarketMaker(),
    ).run()

    return analyze_market_maker(result).summary


def run_ai(seed):
    config = SimulationConfig(
        steps=STEPS,
        seed=seed,
    )

    maker = DirectionalMLMarketMaker(
        **AI_PARAMS
    )

    simulation = MarketSimulation(
        config=config,
        market_maker=maker,
    )

    predictor = DirectionalFlowPredictor()
    cursor = 0

    for timestamp in range(STEPS):
        simulation.step(timestamp)

        new_trades = simulation.trade_rows[cursor:]
        cursor = len(simulation.trade_rows)

        predictor.observe(
            simulation.snapshot_rows[-1],
            new_trades,
        )

        probabilities = predictor.predict_next()

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

    return analyze_market_maker(result).summary


rows = []

for number, seed in enumerate(
    SEEDS,
    start=1,
):
    print(
        f"final market {number}/12 | seed={seed}"
    )

    baseline = run_baseline(seed)
    ai = run_ai(seed)

    rows.append(
        {
            "seed": seed,
            "baseline_pnl": baseline.total_pnl,
            "ai_pnl": ai.total_pnl,
            "pnl_difference": (
                ai.total_pnl
                - baseline.total_pnl
            ),
            "baseline_adverse": (
                baseline.adverse_selection_cost
            ),
            "ai_adverse": (
                ai.adverse_selection_cost
            ),
            "baseline_quantity": (
                baseline.filled_quantity
            ),
            "ai_quantity": (
                ai.filled_quantity
            ),
            "baseline_fill_rate": (
                baseline.fill_rate
            ),
            "ai_fill_rate": (
                ai.fill_rate
            ),
            "baseline_max_inventory": (
                baseline.max_absolute_inventory
            ),
            "ai_max_inventory": (
                ai.max_absolute_inventory
            ),
        }
    )


df = pd.DataFrame(rows)

df.to_csv(
    "results/tables/final_directional_ai_benchmark.csv",
    index=False,
)


difference = df["pnl_difference"]

mean_difference = difference.mean()

standard_error = (
    difference.std(ddof=1)
    / math.sqrt(len(df))
)

ci_low = (
    mean_difference
    - 1.96 * standard_error
)

ci_high = (
    mean_difference
    + 1.96 * standard_error
)

baseline_pnl = df["baseline_pnl"].mean()
ai_pnl = df["ai_pnl"].mean()

baseline_adverse = (
    df["baseline_adverse"].mean()
)

ai_adverse = (
    df["ai_adverse"].mean()
)


print()
print("FINAL AI MARKET-MAKING RESULT")
print("=============================")

print()
print(
    f"Traditional mean P&L: {baseline_pnl:,.2f}"
)

print(
    f"Directional AI P&L:  {ai_pnl:,.2f}"
)

print(
    f"Mean improvement:     {mean_difference:+,.2f}"
)

print(
    f"Median improvement:   {difference.median():+,.2f}"
)

print(
    f"AI win rate:          {(difference > 0).mean():.1%}"
)

print(
    f"95% CI:               [{ci_low:,.2f}, {ci_high:,.2f}]"
)

print()
print("ADVERSE SELECTION")

print(
    f"Traditional:          {baseline_adverse:,.2f}"
)

print(
    f"Directional AI:       {ai_adverse:,.2f}"
)

print(
    f"Change:               {ai_adverse - baseline_adverse:+,.2f}"
)

print()
print("TRADING ACTIVITY")

print(
    f"Traditional quantity: {df['baseline_quantity'].mean():,.1f}"
)

print(
    f"AI quantity:          {df['ai_quantity'].mean():,.1f}"
)

print(
    f"Traditional fill rate: {df['baseline_fill_rate'].mean():.1%}"
)

print(
    f"AI fill rate:          {df['ai_fill_rate'].mean():.1%}"
)

print()
print("INVENTORY RISK")

print(
    f"Traditional max inv:  {df['baseline_max_inventory'].mean():.2f}"
)

print(
    f"AI max inv:           {df['ai_max_inventory'].mean():.2f}"
)

print()
print("AI PARAMETERS")
print(AI_PARAMS)