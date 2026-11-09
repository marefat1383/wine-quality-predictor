import argparse

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model():
    data = load_wine()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )
    labels = np.where(y_train == 0, "standard", "high")
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(x_train, labels)
    return model, x_test, y_test


def predict_quality(index: int = 0) -> tuple[str, str]:
    model, x_test, y_test = build_model()
    if index < 0 or index >= len(x_test):
        raise ValueError(f"Index must be between 0 and {len(x_test) - 1}")
    predicted = str(model.predict([x_test[index]])[0])
    actual = "standard" if y_test[index] == 0 else "high"
    return predicted, actual


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a wine quality category.")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    predicted, actual = predict_quality(args.index)
    print(f"Predicted: {predicted} | Actual: {actual}")


if __name__ == "__main__":
    main()
