from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA = Path(
    "results/tables/directional_flow_dataset.csv"
)

MODEL_DIR = Path("results/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "flow_regime"


def make_logistic():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def make_boosting():
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_iter=300,
                    max_leaf_nodes=20,
                    l2_regularization=1.0,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(model, X, y):
    probability = model.predict_proba(X)
    prediction = model.predict(X)

    auc = roc_auc_score(
        y,
        probability,
        multi_class="ovr",
        average="macro",
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y,
            prediction,
            labels=[0, 1, 2],
            zero_division=0,
        )
    )

    return {
        "auc": auc,
        "accuracy": accuracy_score(
            y,
            prediction,
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                prediction,
            ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


df = pd.read_csv(DATA)

seeds = np.array(
    sorted(
        df["simulation_seed"].unique()
    )
)

rng = np.random.default_rng(42)
rng.shuffle(seeds)

n = len(seeds)

n_train = int(n * 0.70)
n_validation = int(n * 0.15)

train_seeds = seeds[:n_train]

validation_seeds = seeds[
    n_train:
    n_train + n_validation
]

test_seeds = seeds[
    n_train + n_validation:
]

train = df[
    df["simulation_seed"].isin(
        train_seeds
    )
]

validation = df[
    df["simulation_seed"].isin(
        validation_seeds
    )
]

test = df[
    df["simulation_seed"].isin(
        test_seeds
    )
]

models = {
    "Logistic Regression":
        make_logistic(),
    "Gradient Boosting":
        make_boosting(),
}

scores = {}

print()
print("DIRECTIONAL FLOW MODEL")
print("======================")

for name, model in models.items():
    model.fit(
        train[FEATURES],
        train[TARGET],
    )

    score = evaluate(
        model,
        validation[FEATURES],
        validation[TARGET],
    )

    scores[name] = score["auc"]

    print()
    print(name)
    print(
        f"Validation macro ROC-AUC: "
        f"{score['auc']:.3f}"
    )
    print(
        f"Balanced accuracy: "
        f"{score['balanced_accuracy']:.3f}"
    )

best_name = max(
    scores,
    key=scores.get,
)

combined = pd.concat(
    [train, validation],
    ignore_index=True,
)

if best_name == "Logistic Regression":
    model = make_logistic()
else:
    model = make_boosting()

model.fit(
    combined[FEATURES],
    combined[TARGET],
)

score = evaluate(
    model,
    test[FEATURES],
    test[TARGET],
)

print()
print(
    f"Selected model: {best_name}"
)

print()
print("FINAL DIRECTIONAL TEST")
print("======================")

print(
    f"Macro ROC-AUC:      "
    f"{score['auc']:.3f}"
)

print(
    f"Accuracy:           "
    f"{score['accuracy']:.3f}"
)

print(
    f"Balanced accuracy:  "
    f"{score['balanced_accuracy']:.3f}"
)

print()
print("Recall by class:")
print(
    f"Normal:             "
    f"{score['recall'][0]:.3f}"
)
print(
    f"Informed BUY:       "
    f"{score['recall'][1]:.3f}"
)
print(
    f"Informed SELL:      "
    f"{score['recall'][2]:.3f}"
)

artifact = {
    "model": model,
    "features": FEATURES,
    "model_name": best_name,
    "classes": {
        0: "normal",
        1: "buy",
        2: "sell",
    },
}

path = (
    MODEL_DIR
    / "directional_flow_model.joblib"
)

joblib.dump(
    artifact,
    path,
)

print()
print(
    f"Saved model to: {path}"
)
