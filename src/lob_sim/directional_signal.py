import math

import joblib
import numpy as np
import pandas as pd


class DirectionalFlowPredictor:
    def __init__(
        self,
        model_path="results/models/directional_flow_model.joblib",
    ):
        artifact = joblib.load(model_path)

        self.model = artifact["model"]
        self.features = artifact["features"]

        self.mid_prices = []
        self.mid_changes = []

        self.trade_counts = []
        self.volumes = []
        self.signed_flows = []

        self.latest_snapshot = None

    def observe(
        self,
        snapshot,
        trades,
    ):
        self.latest_snapshot = snapshot

        best_bid = snapshot["best_bid"]
        best_ask = snapshot["best_ask"]

        if (
            best_bid is None
            or best_ask is None
        ):
            mid = np.nan
        else:
            mid = (
                best_bid + best_ask
            ) / 2

        if (
            self.mid_prices
            and math.isfinite(mid)
            and math.isfinite(
                self.mid_prices[-1]
            )
        ):
            mid_change = (
                mid
                - self.mid_prices[-1]
            )
        else:
            mid_change = np.nan

        self.mid_prices.append(mid)
        self.mid_changes.append(mid_change)

        trade_count = len(trades)

        volume = sum(
            trade["quantity"]
            for trade in trades
        )

        signed_flow = 0

        for trade in trades:
            if (
                trade["aggressor_side"]
                == "buy"
            ):
                signed_flow += (
                    trade["quantity"]
                )
            else:
                signed_flow -= (
                    trade["quantity"]
                )

        self.trade_counts.append(
            trade_count
        )

        self.volumes.append(
            volume
        )

        self.signed_flows.append(
            signed_flow
        )

    def _mid_change(
        self,
        periods,
    ):
        if (
            len(self.mid_prices)
            <= periods
        ):
            return np.nan

        current = self.mid_prices[-1]
        previous = self.mid_prices[
            -periods - 1
        ]

        if (
            not math.isfinite(current)
            or not math.isfinite(previous)
        ):
            return np.nan

        return current - previous

    def _rolling_sum(
        self,
        values,
        window,
    ):
        return float(
            sum(
                values[-window:]
            )
        )

    def _realized_volatility(
        self,
        window,
    ):
        values = np.array(
            self.mid_changes[-window:],
            dtype=float,
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) < 2:
            return np.nan

        return float(
            np.std(
                values,
                ddof=1,
            )
        )

    def predict_next(
        self,
    ):
        if self.latest_snapshot is None:
            return 0.0

        s = self.latest_snapshot

        row = {
            "spread_lag1":
                s["spread"],

            "volatility_lag1":
                s["volatility"],

            "inventory_lag1":
                s["inventory"],

            "mid_change_1_lag1":
                self._mid_change(1),

            "mid_change_3_lag1":
                self._mid_change(3),

            "mid_change_5_lag1":
                self._mid_change(5),

            "mid_change_10_lag1":
                self._mid_change(10),

            "best_bid_depth_lag1":
                s["best_bid_depth"],

            "best_ask_depth_lag1":
                s["best_ask_depth"],

            "bid_depth_3_lag1":
                s["bid_depth_3"],

            "ask_depth_3_lag1":
                s["ask_depth_3"],

            "book_imbalance_3_lag1":
                s["book_imbalance_3"],
        }

        total_best_depth = (
            s["best_bid_depth"]
            + s["best_ask_depth"]
        )

        if total_best_depth == 0:
            row[
                "best_depth_imbalance_lag1"
            ] = np.nan
        else:
            row[
                "best_depth_imbalance_lag1"
            ] = (
                s["best_bid_depth"]
                - s["best_ask_depth"]
            ) / total_best_depth

        for window in [
            1,
            3,
            5,
            10,
        ]:
            trade_count = (
                self._rolling_sum(
                    self.trade_counts,
                    window,
                )
            )

            volume = (
                self._rolling_sum(
                    self.volumes,
                    window,
                )
            )

            signed_flow = (
                self._rolling_sum(
                    self.signed_flows,
                    window,
                )
            )

            row[
                f"trade_count_{window}"
            ] = trade_count

            row[
                f"volume_{window}"
            ] = volume

            if volume == 0:
                imbalance = np.nan
            else:
                imbalance = (
                    signed_flow
                    / volume
                )

            row[
                f"order_flow_imbalance_{window}"
            ] = imbalance

        row[
            "realized_volatility_5"
        ] = self._realized_volatility(5)

        row[
            "realized_volatility_10"
        ] = self._realized_volatility(10)

        X = pd.DataFrame(
            [row],
            columns=self.features,
        )

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        classes = (
            self.model.classes_
        )

        values = {
            int(label): float(probability)
            for label, probability
            in zip(
                classes,
                probabilities,
            )
        }

        return {
            "normal": values.get(0, 0.0),
            "buy": values.get(1, 0.0),
            "sell": values.get(2, 0.0),
        }
