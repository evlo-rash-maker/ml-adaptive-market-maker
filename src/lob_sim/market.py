from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketState:
    timestamp: int
    fair_value: float
    volatility: float


class FairValueProcess:
    def __init__(
        self,
        initial_value=10_000.0,
        base_volatility=2.0,
        mean_reversion=0.002,
        long_run_value=None,
        seed=42,
    ):
        self.value = float(initial_value)
        self.base_volatility = float(base_volatility)
        self.mean_reversion = float(mean_reversion)

        if long_run_value is None:
            long_run_value = initial_value

        self.long_run_value = float(long_run_value)
        self.rng = np.random.default_rng(seed)

    def advance(
        self,
        timestamp,
        volatility_multiplier=1.0,
        information_shock=0.0,
    ):
        volatility = (
            self.base_volatility
            * volatility_multiplier
        )

        noise = self.rng.normal(
            0.0,
            volatility,
        )

        reversion = (
            self.mean_reversion
            * (
                self.long_run_value
                - self.value
            )
        )

        self.value += (
            noise
            + reversion
            + information_shock
        )

        self.value = max(
            1.0,
            self.value,
        )

        return MarketState(
            timestamp=timestamp,
            fair_value=self.value,
            volatility=volatility,
        )