# Wine Quality Predictor

A small educational AI project that predicts whether a wine sample is likely to have high or standard quality using a Random Forest classifier and the built-in scikit-learn wine dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run

```bash
python wine_quality.py --index 0
```

The index selects an example from the test set and prints the predicted and actual quality category.

## Test

```bash
python -m unittest test_wine_quality.py
```

This project is for learning only and does not assess real wine quality or safety.
