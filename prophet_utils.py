import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

FORECAST_DAYS = 30 * 12


def prophet_forecast(
    df,
    value_col,
    model_path,
    output_path
):
    """
    Generic Prophet inference utility.
    """

    # Resample daily
    daily = df[[value_col]].resample("D").sum()

    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(daily[[value_col]])
    daily[value_col] = scaled

    # Prophet format
    prophet_df = daily.reset_index().rename(
        columns={"Date": "ds", value_col: "y"}
    )

    # Load pretrained model
    model = joblib.load(model_path)

    # Forecast
    future = model.make_future_dataframe(
        periods=FORECAST_DAYS,
        freq="D",
        include_history=False
    )
    forecast = model.predict(future)[["ds", "yhat"]]

    # Inverse scale
    forecast["yhat"] = scaler.inverse_transform(
        forecast[["yhat"]]
    )

    forecast.columns = ["Date", value_col]
    forecast[value_col] = forecast[value_col].abs()

    forecast.to_csv(output_path, index=False)
    return forecast
