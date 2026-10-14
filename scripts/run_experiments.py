from pathlib import Path

import pandas as pd
import yaml

from lob_sim.accounting import analyze_market_maker
from lob_sim.market_maker import MarketMaker
from lob_sim.simulation import MarketSimulation, SimulationConfig


CONFIG_PATH = Path("config/experiments.yaml")
OUTPUT_DIR = Path("results/tables")


def run_once(seed, steps, params):
    maker = MarketMaker(
        base_half_spread=params["base_half_spread"],
        volatility_sensitivity=params["volatility_sensitivity"],
        inventory_skew=params["inventory_skew"],
        max_inventory=params["max_inventory"],
        quote_size=params["quote_size"],
    )

    config = SimulationConfig(
        steps=steps,
        seed=seed,
        informed_trader_share=params["informed_trader_share"],
    )

    result = MarketSimulation(
        config=config,
        market_maker=maker,
    ).run()

    summary = analyze_market_maker(
        result
    ).summary

    return {
        "seed": seed,
        "total_pnl": summary.total_pnl,
        "spread_capture": summary.spread_capture,
        "selection_pnl": summary.selection_pnl,
        "adverse_selection_cost": summary.adverse_selection_cost,
        "inventory_carry_pnl": summary.inventory_carry_pnl,
        "fill_rate": summary.fill_rate,
        "max_absolute_inventory": summary.max_absolute_inventory,
        "trade_count": summary.market_maker_trade_count,
    }


def main():
    with CONFIG_PATH.open() as file:
        config = yaml.safe_load(file)

    steps = config["simulation"]["steps"]
    seeds = config["simulation"]["seeds"]
    baseline = config["baseline"]

    rows = []

    for experiment, settings in config["experiments"].items():
        for value in settings["values"]:
            print(f"{experiment}: {value}")

            for seed in range(seeds):
                params = baseline.copy()

                if experiment == "spread":
                    params["base_half_spread"] = value

                elif experiment == "inventory_limit":
                    params["max_inventory"] = int(value)

                elif experiment == "informed_share":
                    params["informed_trader_share"] = value

                row = run_once(
                    seed=seed,
                    steps=steps,
                    params=params,
                )

                row["experiment"] = experiment
                row["parameter_value"] = value

                rows.append(row)

    raw = pd.DataFrame(rows)

    summary = (
        raw.groupby(
            ["experiment", "parameter_value"]
        )
        .agg(
            pnl_mean=("total_pnl", "mean"),
            pnl_std=("total_pnl", "std"),
            spread_capture_mean=("spread_capture", "mean"),
            selection_pnl_mean=("selection_pnl", "mean"),
            adverse_selection_mean=("adverse_selection_cost", "mean"),
            inventory_pnl_mean=("inventory_carry_pnl", "mean"),
            fill_rate_mean=("fill_rate", "mean"),
            max_inventory_mean=("max_absolute_inventory", "mean"),
            trades_mean=("trade_count", "mean"),
        )
        .reset_index()
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw.to_csv(
        OUTPUT_DIR / "experiment_runs.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "experiment_summary.csv",
        index=False,
    )

    print()
    print("EXPERIMENT SUMMARY")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
