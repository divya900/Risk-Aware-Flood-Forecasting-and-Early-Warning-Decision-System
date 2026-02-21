import pandas as pd
from prophet_utils import prophet_forecast


def daily_runoff_forecast(filename, wtd=1):
    df = pd.read_excel(f"data/{filename}.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    return prophet_forecast(
        df=df,
        value_col="Discharge",
        model_path=f"trained/{filename}_discharge_prophet.pkl",
        output_path=f"data/forecast/{filename}_discharge_forecast.csv"
    )
