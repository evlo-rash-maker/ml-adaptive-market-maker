from lob_sim.accounting import (
    analyze_market_maker,
)
from lob_sim.simulation import (
    MarketSimulation,
    SimulationConfig,
)


config = SimulationConfig(
    steps=1_000,
    seed=42,
)

simulation = MarketSimulation(
    config=config
)

result = simulation.run()

accounting = analyze_market_maker(
    result
)

summary = accounting.summary

print()
print("MARKET MAKER RESULT")
print("-------------------")

print(
    f"Total P&L: "
    f"{summary.total_pnl:,.2f}"
)

print(
    f"Spread capture: "
    f"{summary.spread_capture:,.2f}"
)

print(
    f"Selection P&L: "
    f"{summary.selection_pnl:,.2f}"
)

print(
    f"Adverse selection cost: "
    f"{summary.adverse_selection_cost:,.2f}"
)

print(
    f"Inventory carry P&L: "
    f"{summary.inventory_carry_pnl:,.2f}"
)

print(
    f"MM trades: "
    f"{summary.market_maker_trade_count}"
)

print(
    f"Filled quantity: "
    f"{summary.filled_quantity}"
)

print(
    f"Final inventory: "
    f"{summary.final_inventory}"
)

print(
    f"Max absolute inventory: "
    f"{summary.max_absolute_inventory}"
)

print()