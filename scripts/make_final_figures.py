from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA = Path(
    "results/tables/final_directional_ai_benchmark.csv"
)

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)


# 1. P&L comparison
means = [
    df["baseline_pnl"].mean(),
    df["ai_pnl"].mean(),
]

plt.figure(figsize=(7, 5))
plt.bar(
    ["Traditional", "ML Adaptive"],
    means,
)
plt.ylabel("Mean P&L")
plt.title("Out-of-Sample Market-Making P&L")
plt.tight_layout()
plt.savefig(
    OUT / "final_pnl_comparison.png",
    dpi=200,
)
plt.close()


# 2. P&L improvement by unseen market
plt.figure(figsize=(9, 5))
plt.bar(
    df["seed"].astype(str),
    df["pnl_difference"],
)
plt.axhline(0, linewidth=1)
plt.xlabel("Unseen Market Seed")
plt.ylabel("AI P&L − Traditional P&L")
plt.title("ML Strategy Improvement Across Unseen Markets")
plt.tight_layout()
plt.savefig(
    OUT / "final_pnl_by_market.png",
    dpi=200,
)
plt.close()


# 3. Adverse-selection comparison
adverse = [
    df["baseline_adverse"].mean(),
    df["ai_adverse"].mean(),
]

plt.figure(figsize=(7, 5))
plt.bar(
    ["Traditional", "ML Adaptive"],
    adverse,
)
plt.ylabel("Mean Adverse-Selection Cost")
plt.title("ML Strategy Reduces Adverse Selection")
plt.tight_layout()
plt.savefig(
    OUT / "final_adverse_selection.png",
    dpi=200,
)
plt.close()


# 4. Trade-off summary
summary = pd.DataFrame(
    {
        "Traditional": [
            df["baseline_fill_rate"].mean() * 100,
            df["baseline_max_inventory"].mean(),
        ],
        "ML Adaptive": [
            df["ai_fill_rate"].mean() * 100,
            df["ai_max_inventory"].mean(),
        ],
    },
    index=[
        "Fill Rate (%)",
        "Max Inventory",
    ],
)

summary.plot(
    kind="bar",
    figsize=(8, 5),
)
plt.xticks(rotation=0)
plt.title("Execution and Inventory Trade-Off")
plt.tight_layout()
plt.savefig(
    OUT / "final_risk_tradeoff.png",
    dpi=200,
)
plt.close()


print("Final figures created:")
for path in sorted(OUT.glob("*.png")):
    print(path)
