import pandas as pd
import numpy as np
from datetime import datetime

import discharge_prophet as dp
import flood_runoff_prophet as frp
import daily_runoff_prophet as drp
import weekly_runoff_prophet as wrp
import model

DATA_PATH = "data/"


# ----------------------------
# Utility Functions
# ----------------------------

def is_future_date(user_date, last_date):
    return user_date > last_date


def clean_dataframe(df):
    for col in df.columns:
        if col != "Date":
            df[col] = df[col].fillna(df[col].mean())
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def format_result(discharge, floodrunoff, dailyrunoff, weeklyrunoff,
                  mae, predicted, actual):
    return {
        "discharge": round(float(discharge), 2),
        "floodrunoff": round(float(floodrunoff), 2),
        "dailyrunoff": round(float(dailyrunoff), 2),
        "weeklyrunoff": round(float(weeklyrunoff), 2),
        "mae": mae,
        "predicted": predicted,
        "actual": actual
    }


# ----------------------------
# Main Driver Function
# ----------------------------

def drive(river_name, user_date):
    """
    Core flood prediction driver.
    Handles:
    - Past date analysis
    - Future date forecasting
    """

    user_date = pd.to_datetime(user_date)

    # Load river dataset
    data = pd.read_excel(f"{DATA_PATH}{river_name}.xlsx")
    data = clean_dataframe(data)

    last_date = data["Date"].iloc[-1]

    # ----------------------------
    # PAST DATE ANALYSIS
    # ----------------------------
    if not is_future_date(user_date, last_date):

        row = data[data["Date"] == user_date]

        if row.empty:
            return {"error": "Invalid or unavailable past date"}

        discharge = row["Discharge"].values[0]
        floodrunoff = row["flood runoff"].values[0]
        dailyrunoff = row["daily runoff"].values[0]
        weeklyrunoff = row["weekly runoff"].values[0]
        actual_flood = row["Flood"].values[0]

        features = [discharge, floodrunoff, dailyrunoff, weeklyrunoff]

        result, mae = model.flood_classifier(river_name, features)

        predicted = "High" if result == 1 else "Normal"
        actual = "High" if actual_flood == 1 else "Normal"

        return format_result(
            discharge, floodrunoff, dailyrunoff, weeklyrunoff,
            round(mae, 2), predicted, actual
        )

    # ----------------------------
    # FUTURE DATE PREDICTION
    # ----------------------------
    else:
        wtd = 1  # forecast mode

        d1 = dp.discharge_forecast(river_name, wtd)
        d2 = frp.flood_runoff_forecast(river_name, wtd)
        d3 = drp.daily_runoff_forecast(river_name, wtd)
        d4 = wrp.weekly_runoff_forecast(river_name, wtd)

        future_df = pd.concat(
            [d1, d2["flood runoff"], d3["daily runoff"], d4["weekly runoff"]],
            axis=1
        )
        future_df["Date"] = pd.to_datetime(future_df["Date"])

        row = future_df[future_df["Date"] == user_date]

        if row.empty:
            return {"error": "Invalid or unavailable future date"}

        discharge = row["Discharge"].values[0]
        floodrunoff = row["flood runoff"].values[0]
        dailyrunoff = row["daily runoff"].values[0]
        weeklyrunoff = row["weekly runoff"].values[0]

        features = [discharge, floodrunoff, dailyrunoff, weeklyrunoff]

        result, _ = model.flood_classifier(river_name, features)
        predicted = "High" if result == 1 else "Normal"

        return format_result(
            discharge, floodrunoff, dailyrunoff, weeklyrunoff,
            "N/A", predicted, "N/A"
        )
