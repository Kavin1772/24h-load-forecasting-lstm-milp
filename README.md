# 24-Hour Ahead Residential Load Forecasting and Scheduling Using LSTM and MILP

## 📌 Overview
This project presents an end-to-end pipeline for short-term residential electricity load forecasting and appliance scheduling. A Long Short-Term Memory (LSTM) neural network is used to predict the next 24 hours of household power consumption, and the forecast is then integrated into a Mixed-Integer Linear Programming (MILP) formulation to schedule flexible appliances under cost and peak-load constraints.

The system is evaluated using real residential data from the UK-DALE dataset.

## 🎯 Objectives
- Forecast 24-hour ahead residential electricity demand
- Study the effect of LSTM hyperparameters on prediction accuracy
- Optimize appliance scheduling using MILP
- Provide a fully reproducible research pipeline

## 📊 Dataset
- **UK-DALE (House 1)**
- Hourly aggregated active power consumption
- Sliding window formulation:
  - Input window: 168 hours
  - Forecast horizon: 24 hours

## 🧠 Methodology
### Forecasting
- Baseline models:
  - Linear Regression
  - Random Forest
- Deep learning model:
  - Encoder-style LSTM (PyTorch)
  - Hyperparameter study on:
    - Hidden size
    - Number of layers
    - Dropout rate

### Scheduling
- MILP formulation for appliance scheduling
- Time-of-use tariff
- Comparison between naive and optimized schedules

## 🗂️ Project Structure
project/
│── Data File/
│   ├── ukdale_house1.csv
│   └── ukdale.h5
│── src/
│   ├── data.py
│   ├── features_lstm.py
│   ├── models/
│   │   └── lstm.py
│   └── scheduling/
│       ├── heuristics.py
│       └── milp_scheduler.py
│── scripts/
│   ├── train_lstm.py
│   ├── run_scheduler.py
│   └── run_milp_scheduler.py
│── configs/
│   ├── config.yaml
│   └── tariff_and_appliances.yaml
│── artifacts/
│── outputs/
│── README.md
│── requirements.txt

