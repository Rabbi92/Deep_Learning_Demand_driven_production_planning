# Demand-Driven Production Scheduling System — Company Y (National Foods)

A deep-learning demand forecasting + cost-aware production scheduling decision-support
system for a multi-product grain mill, delivered as a Streamlit web application.

This is the deployment artefact of an MEng dissertation. The model is trained in the
companion Kaggle notebook; this app loads the trained model and reproduces the
forecast → schedule → cost workflow in the browser.

## What the app does

- **Forecasts** 7-day product-level demand (Maize Meal, Flour, Rice, Pasta) using a
  global LSTM trained on true (uncensored) demand.
- **Schedules** production with an industrial-engineering cost model that minimises
  changeover + holding + stockout cost, subject to capacity, inventory-band,
  shelf-life and service-level constraints.
- Offers two solvers: a **greedy IE heuristic** and a **genetic algorithm**.
- Compares the optimised plan against a **reactive baseline** and reports the saving.

Embedded IE rules: demand-driven lot sizing · sequence-dependent changeover
minimisation · capacity balancing · inventory & shelf-life control · service-level protection.

## Repository structure

```
.
├── app.py                                  # Streamlit application
├── requirements.txt                        # dependencies
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml                         # theme
├── lstm_demand.keras                        # trained model  (from Kaggle Output tab)
├── xscaler.joblib                           # feature scaler  (from Kaggle)
├── yscaler.joblib                           # target scaler   (from Kaggle)
├── config.joblib                            # IE parameters   (from Kaggle)
└── nationalfoods_demand_2023_2025.csv       # demand history
```

## How to deploy

1. Run the Kaggle notebook and download from its **Output** tab:
   `lstm_demand.keras`, `xscaler.joblib`, `yscaler.joblib`, `config.joblib`.
2. Place those four files + `nationalfoods_demand_2023_2025.csv` in this repo root.
3. Push the repo to GitHub.
4. On https://share.streamlit.io , create a new app pointing at `app.py` on your repo.
   Streamlit Community Cloud builds from `requirements.txt` and serves a public URL.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs even without the model files (a seasonal-naive fallback forecaster keeps the
UI functional for demonstration), but for real forecasts include the trained `.keras` model.

## Data note

The dataset is **synthetic, calibrated** to a documented expert interview and public sources
(National Foods Annual Report 2024; Zimbabwe school/holiday calendars; the 2023/24 El Niño
drought). Forecasting accuracy is validated on this calibrated history; cost savings are
presented as projected/indicative.
