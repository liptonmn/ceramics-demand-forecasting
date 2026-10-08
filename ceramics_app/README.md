# Ceramics Weekly Demand Forecast App

A FastAPI app that serves the champion model from the MLflow Model Registry.
Enter an item number (1-50) to get its total weekly demand across all 10 stores
for the next 4 weeks.

## What's in this folder

- `app.py` - the FastAPI app
- `mlflow.db` - MLflow tracking and registry database (knows which model is champion)
- `mlruns/` - the saved trained models
- `processed_data/` - the versioned weekly sales data (Parquet)
- `requirements.txt` - the exact package versions the app was tested with

## How to run it

Requires Python 3.10 or newer.

1. Download or clone this repository.
2. Open a terminal in this `ceramics_app` folder.
3. Install the required packages:

```
   pip install -r requirements.txt
```

4. Start the app:

```
   python -m uvicorn app:app
```

5. Open a browser and go to:
   - http://127.0.0.1:8000 for the forecast page
   - http://127.0.0.1:8000/docs for the API documentation

## Notes

- The model files were produced by running the project notebook. To use a newly
  trained champion, rerun the notebook and replace `mlflow.db`, `mlruns/`, and
  `processed_data/` with the new versions.
- On startup, MLflow may print warnings about package or Python version
  differences from the training environment. These do not affect the forecasts.