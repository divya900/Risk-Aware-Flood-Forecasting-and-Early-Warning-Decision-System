import pandas as pd
from datetime import datetime

import discharge_prophet as dp
import flood_runoff_prophet as frp
import daily_runoff_prophet as drp
import weekly_runoff_prophet as wrp
import model

DATA_PATH = "data/"
RIVERS = ["Cauvery", "Godavari", "Krishna", "Mahanadi", "Son"]


# ------------------------------------------------
# Flood Alert Reader (Used by UI)
# ------------------------------------------------

def alerting():
    """
    Reads precomputed flood alerts and returns
    rivers likely to flood.
    """
    try:
        df = pd.read_csv("data/forecast/forecasted_level_of_rivers.csv")
    except FileNotFoundError:
        return []

    alert_rivers = []
    for river in df.columns:
        if df[river].max() == 1:
            alert_rivers.append(river)

    return alert_rivers


# ------------------------------------------------
# Flood Alert Generator (12-month forecast)
# ------------------------------------------------

def water_level_predictior():
    """
    Generates flood alerts for next 12 months
    using Prophet + LDA.
    """

    predictions = {}

    for river in RIVERS:

        # Forecast future features
        d1 = dp.discharge_forecast(river, wtd=1)
        d2 = frp.flood_runoff_forecast(river, wtd=1)
        d3 = drp.daily_runoff_forecast(river, wtd=1)
        d4 = wrp.weekly_runoff_forecast(river, wtd=1)

        future_df = pd.concat(
            [
                d1,
                d2["flood runoff"],
                d3["daily runoff"],
                d4["weekly runoff"]
            ],
            axis=1
        )

        river_predictions = []

        for _, row in future_df.iterrows():
            features = [
                row["Discharge"],
                row["flood runoff"],
                row["daily runoff"],
                row["weekly runoff"]
            ]

            result, _ = model.flood_classifier(river, features)
            river_predictions.append(int(result))

        predictions[river] = river_predictions

    alert_df = pd.DataFrame(predictions)
    alert_df.to_csv(
        "data/forecast/forecasted_level_of_rivers.csv",
        index=False
    )

    return
