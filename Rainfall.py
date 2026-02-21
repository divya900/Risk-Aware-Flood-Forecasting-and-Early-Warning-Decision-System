import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, explained_variance_score

DATA_PATH = "data/Sub_Division_IMD_2017.csv"
MODEL_PATH = "models/rainfall_model.h5"


# ------------------------------------------------
# Feature Engineering
# ------------------------------------------------

def generate_features_by_year(data, year, region):
    temp = data[data["YEAR"] == year]
    temp = temp[temp["SUBDIVISION"] == region]

    values = temp[
        ["JAN","FEB","MAR","APR","MAY","JUN",
         "JUL","AUG","SEP","OCT","NOV","DEC"]
    ].values

    X, y = [], []
    for i in range(values.shape[1] - 3):
        X.append(values[:, i:i+3])
        y.append(values[:, i+3])

    return np.vstack(X), np.hstack(y)


# ------------------------------------------------
# Main Backend Function
# ------------------------------------------------

def rainfall(year, region):
    """
    Backend rainfall prediction.
    Returns MAE and explained variance score.
    """

    data = pd.read_csv(DATA_PATH)
    data = data.fillna(data.mean(numeric_only=True))

    model = load_model(MODEL_PATH)

    X_test, y_test = generate_features_by_year(data, int(year), region)

    if X_test.size == 0:
        return "N/A", "N/A"

    X_test = np.expand_dims(X_test, axis=2)

    y_pred = model.predict(X_test, verbose=0)

    mae = mean_absolute_error(y_test, y_pred)
    score = explained_variance_score(y_test, y_pred)

    return round(float(mae), 2), round(float(score), 2)
