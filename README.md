# Risk-Aware-Flood-Forecasting-and-Early-Warning-Decision-System

> An interpretable, probabilistic flood risk assessment system integrating rainfall forecasting, river flow prediction, and multi-horizon early warning alerts using hybrid machine learning models.

## Project Overview

Floods are high-impact disasters where **early detection and reliable risk estimation** are critical.
This project implements a **hybrid hydrological forecasting system** that:

* Forecasts regional rainfall
* Predicts river discharge and runoff
* Generates probabilistic flood risk levels
* Provides explainable early warning alerts

Unlike traditional binary flood prediction systems, this project focuses on:

* Minimizing false negatives
* Probability-based risk estimation
* Model interpretability
* Decision-support orientation


## System Architecture

The system is modular and consists of four main components:

### 1️. Rainfall Forecasting

* Deep learning model (Conv1D architecture)
* Sliding-window feature engineering
* Subdivision-level rainfall prediction
* Evaluated using MAE & explained variance score

### 2️. River Flow Forecasting

* Prophet-based time-series forecasting
* Predicts:

  * Discharge
  * Flood runoff
  * Daily runoff
  * Weekly runoff
* 12-month forecast horizon

### 3️. Flood Risk Inference

* Linear Discriminant Analysis (LDA)
* Outputs:

  * Flood probability
  * Risk level (Low / Medium / High)
* Designed to reduce false-negative risk

### 4️. Early Warning & Backend Service

* Flask-based backend
* Multi-river risk monitoring
* 12-month early warning alerts
* Explainable risk drivers

---

## Modeling Approach

| Component                 | Model                  | Rationale                           |
| ------------------------- | ---------------------- | ----------------------------------- |
| Rainfall Forecasting      | Conv1D (Deep Learning) | Captures temporal rainfall patterns |
| River Flow Forecasting    | Prophet                | Handles seasonality & trend shifts  |
| Flood Risk Classification | LDA                    | Interpretable & probabilistic       |

---

## Explainability & Risk Interpretation

Flood risk predictions are interpretable via:

* LDA model coefficients
* Ranked hydrological risk drivers
* Probability-based outputs

Example Output:

```
Flood Probability: 78.4%
Risk Level: High
Primary Risk Drivers:
  - Discharge
  - Flood Runoff
```


## Model Evaluation

Flood classification evaluated using:

* Confusion Matrix
* False-negative analysis
* MAE (where applicable)

Priority: Reduce missed flood events (false negatives)

Rainfall forecasting evaluated using:

* Mean Absolute Error (MAE)
* Explained Variance Score


## Project Structure

```
.
├── main.py
├── driver.py
├── alerter.py
├── Rainfall.py
├── discharge_prophet.py
├── flood_runoff_prophet.py
├── daily_runoff_prophet.py
├── weekly_runoff_prophet.py
├── data/
├── models/
├── trained/
├── templates/
├── static/
└── requirements.txt
```

---

## Installation

### 1️. Create Virtual Environment

**Windows (PowerShell)**

```powershell
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
```

**Mac/Linux**

```bash
python3.11 -m venv venv
source venv/bin/activate
```


### 2️. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## Run the Application

```bash
python main.py
```

Access at:

```
http://127.0.0.1:5000/
```

## Supported Rivers

* Godavari
* Krishna
* Mahanadi
* Cauvery
* Son


## Limitations

* Uses historical datasets (no real-time sensor integration)
* Limited geographic coverage
* Climate variability may affect long-term forecasting

## Future Enhancements

* Real-time IoT rainfall integration
* Satellite imagery-based hydrological features
* LSTM/Transformer time-series baselines
* Drift detection & automated retraining
* Cloud deployment & REST API expansion

## Key Concepts Demonstrated

* Hybrid ML architecture
* Probabilistic classification
* Risk-aware decision systems
* Time-series forecasting
* Model interpretability
* Modular backend engineering

## License

Academic and research use.

---

## 👤 Author

**Divya**
