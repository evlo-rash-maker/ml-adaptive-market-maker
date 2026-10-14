from pathlib import Path

import numpy as np
import pandas as pd

from lob_sim.simulation import (
    MarketSimulation,
    SimulationConfig,
)


OUTPUT = Path(
    "results/tables/directional_flow_dataset.csv"
)

SEEDS = 40
STEPS = 750

INFORMED_LEVELS = [
    0.05,
    0.15,
    0.30,
    0.50,
]

VOLATILITY_LEVELS = [
    1.5,
    2.0,
    3.0,
]


all_rows = []


for seed in range(SEEDS):
    informed_share = INFORMED_LEVELS[
        seed % len(INFORMED_LEVELS)
    ]

    base_volatility = VOLATILITY_LEVELS[
        seed % len(VOLATILITY_LEVELS)
    ]

    print(
        f"seed {seed + 1}/{SEEDS} "
        f"| informed={informed_share:.0%} "
        f"| vol={base_volatility}"
    )

    config = SimulationConfig(
        steps=STEPS,
        seed=seed,
        informed_trader_share=informed_share,
        base_volatility=base_volatility,
    )

    result = MarketSimulation(
        config=config
    ).run()

    snapshots = (
        result.snapshots
        .copy()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    trades = result.trades.copy()

    # ---------------------------------
    # Hidden answer the ML must predict
    # ---------------------------------

    snapshots["informed_regime"] = (
        snapshots[
            "information_shock"
        ].abs() > 0
    ).astype(int)

    snapshots["flow_regime"] = np.select(
        [
            snapshots["information_shock"] > 0,
            snapshots["information_shock"] < 0,
        ],
        [
            1,
            2,
        ],
        default=0,
    ).astype(int)

    # ---------------------------------
    # Observable market prices
    # ---------------------------------

    snapshots["mid_price"] = (
        snapshots["best_bid"]
        + snapshots["best_ask"]
    ) / 2

    snapshots["mid_change_1"] = (
        snapshots[
            "mid_price"
        ].diff()
    )

    snapshots["mid_change_3"] = (
        snapshots[
            "mid_price"
        ].diff(3)
    )

    snapshots["mid_change_5"] = (
        snapshots[
            "mid_price"
        ].diff(5)
    )

    snapshots["mid_change_10"] = (
        snapshots[
            "mid_price"
        ].diff(10)
    )

    # Only information available
    # BEFORE the current step.

    snapshots["spread_lag1"] = (
        snapshots[
            "spread"
        ].shift(1)
    )

    snapshots["volatility_lag1"] = (
        snapshots[
            "volatility"
        ].shift(1)
    )

    snapshots["inventory_lag1"] = (
        snapshots[
            "inventory"
        ].shift(1)
    )

    snapshots["mid_change_1_lag1"] = (
        snapshots[
            "mid_change_1"
        ].shift(1)
    )

    snapshots["mid_change_3_lag1"] = (
        snapshots[
            "mid_change_3"
        ].shift(1)
    )

    snapshots["mid_change_5_lag1"] = (
        snapshots[
            "mid_change_5"
        ].shift(1)
    )

    snapshots["mid_change_10_lag1"] = (
        snapshots[
            "mid_change_10"
        ].shift(1)
    )

    # ---------------------------------
    # Observable trade flow
    # ---------------------------------

    if trades.empty:
        snapshots["trade_count"] = 0
        snapshots["volume"] = 0
        snapshots["signed_flow"] = 0

    else:
        trades["signed_flow"] = np.where(
            trades[
                "aggressor_side"
            ].eq("buy"),
            trades["quantity"],
            -trades["quantity"],
        )

        flow = (
            trades.groupby(
                "timestamp",
                as_index=False,
            )
            .agg(
                trade_count=(
                    "trade_id",
                    "count",
                ),
                volume=(
                    "quantity",
                    "sum",
                ),
                signed_flow=(
                    "signed_flow",
                    "sum",
                ),
            )
        )

        snapshots = snapshots.merge(
            flow,
            on="timestamp",
            how="left",
        )

        snapshots[
            [
                "trade_count",
                "volume",
                "signed_flow",
            ]
        ] = snapshots[
            [
                "trade_count",
                "volume",
                "signed_flow",
            ]
        ].fillna(0)

    for window in [
        1,
        3,
        5,
        10,
    ]:
        snapshots[
            f"trade_count_{window}"
        ] = (
            snapshots[
                "trade_count"
            ]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .shift(1)
        )

        snapshots[
            f"volume_{window}"
        ] = (
            snapshots[
                "volume"
            ]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .shift(1)
        )

        snapshots[
            f"signed_flow_{window}"
        ] = (
            snapshots[
                "signed_flow"
            ]
            .rolling(
                window,
                min_periods=1,
            )
            .sum()
            .shift(1)
        )

        snapshots[
            f"order_flow_imbalance_{window}"
        ] = (
            snapshots[
                f"signed_flow_{window}"
            ]
            / snapshots[
                f"volume_{window}"
            ].replace(
                0,
                np.nan,
            )
        )

    # ---------------------------------
    # Recent realised volatility
    # ---------------------------------

    snapshots[
        "realized_volatility_5"
    ] = (
        snapshots[
            "mid_change_1"
        ]
        .rolling(
            5,
            min_periods=2,
        )
        .std()
        .shift(1)
    )

    snapshots[
        "realized_volatility_10"
    ] = (
        snapshots[
            "mid_change_1"
        ]
        .rolling(
            10,
            min_periods=2,
        )
        .std()
        .shift(1)
    )

    # ---------------------------------
    # Order-book pressure
    # ---------------------------------

    for column in [
        "best_bid_depth",
        "best_ask_depth",
        "bid_depth_3",
        "ask_depth_3",
        "book_imbalance_3",
    ]:
        snapshots[
            f"{column}_lag1"
        ] = (
            snapshots[
                column
            ].shift(1)
        )

    total_best_depth = (
        snapshots[
            "best_bid_depth_lag1"
        ]
        + snapshots[
            "best_ask_depth_lag1"
        ]
    )

    snapshots[
        "best_depth_imbalance_lag1"
    ] = (
        (
            snapshots[
                "best_bid_depth_lag1"
            ]
            - snapshots[
                "best_ask_depth_lag1"
            ]
        )
        / total_best_depth.replace(
            0,
            np.nan,
        )
    )

    snapshots[
        "simulation_seed"
    ] = seed

    all_rows.append(
        snapshots
    )


dataset = pd.concat(
    all_rows,
    ignore_index=True,
)


FEATURES = [
    "spread_lag1",
    "volatility_lag1",
    "inventory_lag1",

    "mid_change_1_lag1",
    "mid_change_3_lag1",
    "mid_change_5_lag1",
    "mid_change_10_lag1",

    "trade_count_1",
    "trade_count_3",
    "trade_count_5",
    "trade_count_10",

    "volume_1",
    "volume_3",
    "volume_5",
    "volume_10",

    "order_flow_imbalance_1",
    "order_flow_imbalance_3",
    "order_flow_imbalance_5",
    "order_flow_imbalance_10",

    "realized_volatility_5",
    "realized_volatility_10",

    "best_bid_depth_lag1",
    "best_ask_depth_lag1",
    "bid_depth_3_lag1",
    "ask_depth_3_lag1",
    "book_imbalance_3_lag1",
    "best_depth_imbalance_lag1",
]


columns = [
    "simulation_seed",
    "timestamp",
    *FEATURES,
    "informed_regime",
    "flow_regime",
]


dataset = (
    dataset[
        columns
    ]
    .dropna()
    .reset_index(
        drop=True
    )
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


dataset.to_csv(
    OUTPUT,
    index=False,
)


active = int(
    dataset[
        "informed_regime"
    ].sum()
)

inactive = int(
    (
        dataset[
            "informed_regime"
        ] == 0
    ).sum()
)


print()
print(
    "INFORMED REGIME DATASET COMPLETE"
)
print(
    "--------------------------------"
)
print(
    f"Rows: {len(dataset):,}"
)
print(
    f"Informed-regime rows: "
    f"{active:,}"
)
print(
    f"Normal rows: "
    f"{inactive:,}"
)
print(
    f"Informed-regime rate: "
    f"{dataset['informed_regime'].mean():.1%}"
)
print()
print(
    f"Saved to: {OUTPUT}"
)
print()
print("DIRECTIONAL CLASSES")
print("-------------------")
print("0 = normal")
print("1 = informed buying")
print("2 = informed selling")
print()
print(
    dataset["flow_regime"]
    .value_counts()
    .sort_index()
)
