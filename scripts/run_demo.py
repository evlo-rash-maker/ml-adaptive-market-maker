import json

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


SEED = 100
STEPS = 600


with open(
    "results/models/directional_strategy.json"
) as file:
    AI_PARAMS = json.load(file)


def run_baseline():
    result = MarketSimulation(
        config=SimulationConfig(
            steps=STEPS,
            seed=SEED,
        ),
        market_maker=MarketMaker(),
    ).run()

    return analyze_market_maker(result).summary


def run_ai():
    maker = DirectionalMLMarketMaker(
        **AI_PARAMS
    )

    simulation = MarketSimulation(
        config=SimulationConfig(
            steps=STEPS,
            seed=SEED,
        ),
        market_maker=maker,
    )

    predictor = DirectionalFlowPredictor()
    cursor = 0

    probabilities = {
        "normal": 1.0,
        "buy": 0.0,
        "sell": 0.0,
    }

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

    summary = analyze_market_maker(result).summary

    return summary, probabilities


print()
print("ML-ADAPTIVE MARKET MAKING - QUICK DEMO")
print("=======================================")

print()
print(
    f"Running synthetic market "
    f"(seed={SEED}, {STEPS} events)..."
)

baseline = run_baseline()
ai, probabilities = run_ai()

pnl_change = (
    ai.total_pnl
    - baseline.total_pnl
)

adverse_change = (
    ai.adverse_selection_cost
    - baseline.adverse_selection_cost
)


print()
print("STRATEGY COMPARISON")
print("-------------------")

print(
    f"{'Metric':<24}"
    f"{'Traditional':>14}"
    f"{'ML Adaptive':>14}"
)

print(
    f"{'P&L':<24}"
    f"{baseline.total_pnl:>14,.2f}"
    f"{ai.total_pnl:>14,.2f}"
)

print(
    f"{'Adverse selection':<24}"
    f"{baseline.adverse_selection_cost:>14,.2f}"
    f"{ai.adverse_selection_cost:>14,.2f}"
)

print(
    f"{'Filled quantity':<24}"
    f"{baseline.filled_quantity:>14,.1f}"
    f"{ai.filled_quantity:>14,.1f}"
)

print(
    f"{'Fill rate':<24}"
    f"{baseline.fill_rate:>13.1%}"
    f"{ai.fill_rate:>13.1%}"
)

print(
    f"{'Max inventory':<24}"
    f"{baseline.max_absolute_inventory:>14.2f}"
    f"{ai.max_absolute_inventory:>14.2f}"
)


print()
print("FINAL ML STATE")
print("--------------")

print(
    f"P(normal):          "
    f"{probabilities['normal']:.1%}"
)

print(
    f"P(informed buy):    "
    f"{probabilities['buy']:.1%}"
)

print(
    f"P(informed sell):   "
    f"{probabilities['sell']:.1%}"
)


print()
print("DEMO RESULT")
print("-----------")

print(
    f"P&L difference:             "
    f"{pnl_change:+,.2f}"
)

print(
    f"Adverse-selection change:   "
    f"{adverse_change:+,.2f}"
)

print()
print(
    "This is one synthetic market only. "
    "Single-run results can vary."
)

print(
    "See the 12-market held-out benchmark "
    "for the final evaluation."
)

print(
    "This project does not imply "
    "real-world trading profitability."
)

benchmark = pd.read_csv(
    "results/tables/final_directional_ai_benchmark.csv"
)

benchmark_difference = (
    benchmark["ai_pnl"]
    - benchmark["baseline_pnl"]
)

baseline_mean = benchmark[
    "baseline_pnl"
].mean()

ai_mean = benchmark[
    "ai_pnl"
].mean()

relative_improvement = (
    (ai_mean - baseline_mean)
    / baseline_mean
)

adverse_reduction = (
    1
    - (
        benchmark["ai_adverse"].mean()
        / benchmark["baseline_adverse"].mean()
    )
)

print()
print("FULL HELD-OUT BENCHMARK")
print("-----------------------")

print(
    f"Markets tested:              "
    f"{len(benchmark)}"
)

print(
    f"Traditional mean P&L:        "
    f"{baseline_mean:,.2f}"
)

print(
    f"ML mean P&L:                 "
    f"{ai_mean:,.2f}"
)

print(
    f"Mean P&L improvement:        "
    f"{relative_improvement:+.1%}"
)

print(
    f"ML win rate:                 "
    f"{(benchmark_difference > 0).mean():.1%}"
)

print(
    f"Adverse-selection reduction: "
    f"{adverse_reduction:.1%}"
)

print()
print(
    "The single run above demonstrates the system. "
    "The held-out benchmark is the actual evaluation."
)