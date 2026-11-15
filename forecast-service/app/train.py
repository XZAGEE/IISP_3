"""Train the model and persist the resulting artifact."""

from pathlib import Path

import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .model import (
    CAT,
    NUM,
    TARGET,
    load_training_data,
    train_model,
)


ARTIFACTS = (
    Path(__file__).resolve().parent.parent
    / "artifacts"
)


def main() -> None:
    """Train the model, report metrics, and persist the artifact."""

    df = load_training_data()

    X = df[NUM + CAT]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    train_df = X_train.copy()
    train_df[TARGET] = y_train

    model = train_model(train_df)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    print(
        f"[train] Train: R2={train_r2:.4f}, "
        f"MAE={train_mae:.0f} USD"
    )

    print(
        f"[train] Test: R2={test_r2:.4f}, "
        f"MAE={test_mae:.0f} USD"
    )

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_path = ARTIFACTS / "model.pkl"

    final_model = train_model(df)

    joblib.dump(
        final_model,
        artifact_path,
    )

    print(
        f"[train] artifact persisted: {artifact_path}"
    )


if __name__ == "__main__":
    main()
