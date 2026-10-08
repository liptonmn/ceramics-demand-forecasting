# ============================================================
# app.py - Ceramics demand forecast app
# ============================================================

from pathlib import Path          # helps the app find files on your computer
import mlflow                     # the tool that stores and loads our models
from mlflow import MlflowClient   # lets the app ask MLflow's catalog questions


# ---- 3a. Point to the project files ----
# Like writing three addresses on a sticky note.
# Nothing gets opened yet; the app just learns where things are.

# Find the folder this app.py file lives in ("ceramics app").
# Everything else sits in the same folder, so we start from here.
BASE_DIR = Path(__file__).resolve().parent

# The three things you downloaded from Colab:
DB_PATH = BASE_DIR / "mlflow.db"                                    # the catalog (which model is champion)
MLRUNS_DIR = BASE_DIR / "mlruns"                                    # the saved models themselves
DATA_PATH = BASE_DIR / "processed_data" / "weekly_features.parquet" # the weekly sales history

# The name the notebook gave our model in the catalog
MODEL_NAME = "ceramics_weekly_demand"

# Tell MLflow: "the catalog is the mlflow.db file on THIS laptop"
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")


def local_path(colab_address):
    """The catalog still has old Colab addresses like '/content/mlruns/...'.
    This swaps '/content/' for our folder, so the app looks on this laptop."""
    return BASE_DIR / colab_address.replace("/content/", "", 1)


# ---- 3b. Load the champion ----
# Now the app actually opens things: it asks the catalog which model
# is the champion, finds that model's files, and loads it.

# Open a connection to the catalog so we can ask it questions
client = MlflowClient()

# Ask: "Which version has the champion label?" (right now: version 1, XGBoost)
champ = client.get_model_version_by_alias(MODEL_NAME, "champion")

# The catalog lists the model as "models:/m-1aadcf..."
# Keep only the ID part after the last "/"
model_id = champ.source.split("/")[-1]

# Ask where that model was saved. This gives back the OLD Colab address...
colab_address = client.get_logged_model(model_id).artifact_location

# ...so fix it to point at our mlruns folder instead
model_dir = local_path(colab_address)

# Load the actual trained model so it's ready to make predictions
champion_model = mlflow.pyfunc.load_model(str(model_dir))

# Print a message so we can see it worked
print(f"Loaded champion: version {champ.version} from {model_dir}")

import numpy as np    # math helper (we use it for averages)
import pandas as pd   # table helper (reads the sales file, builds rows)


# ---- 3c. Build the features ----
# The model can't take "item 10" directly. It needs the same inputs
# it was trained on: calendar facts + recent sales + which product.
# This part gathers each product's sales history and builds those inputs.

# The last day in the original sales data (train.csv ends Dec 31, 2017).
# The notebook ignores the final, half-finished week after this date,
# so we do the same, so our forecasts match the notebook's.
LAST_RAW_DATE = pd.Timestamp("2017-12-31")

# Open the weekly sales table (the Parquet file)
weekly = pd.read_parquet(DATA_PATH)

# Keep only full weeks
history = weekly[weekly["week_start"] <= LAST_RAW_DATE]

# The most recent full week in the data. Forecasts start the week after this.
LAST_WEEK = history["week_start"].max()

# Each product's weekly sales as a simple list, oldest to newest
# Example: SALES_HISTORY["item_10"] = [3120.0, 3240.0, ..., 4790.0]
SALES_HISTORY = {
    product: group.sort_values("week_start")["units_sold"].astype(float).tolist()
    for product, group in history.groupby("product_category")
}

# Ask the model which input columns it expects, in what order and type.
# (It remembers this from training, so we don't have to guess.)
schema = champion_model.metadata.get_input_schema()
MODEL_COLUMNS = schema.input_names()
MODEL_TYPES = {col.name: col.type.to_numpy() for col in schema.inputs}


def build_features(product, week, sales):
    """Build one row of model inputs for one product and one week.
    'sales' is that product's sales list, newest at the end."""
    row = {
        # Calendar facts about the week we're forecasting
        "week_of_year": int(week.isocalendar().week),
        "month": week.month,
        "is_holiday_season": int(week.month in (11, 12)),
        # Recent sales
        "lag_1": sales[-1],                    # last week
        "lag_4": sales[-4],                    # 4 weeks ago
        "rolling_mean_4": np.mean(sales[-4:]), # average of last 4 weeks
        "rolling_mean_8": np.mean(sales[-8:]), # average of last 8 weeks
        # Which product: a 1 in this product's column (all others become 0)
        f"product_{product}": 1,
    }
    # Line the row up with the exact columns the model expects,
    # fill any missing ones (the other 49 products) with 0,
    # and match the data types the model was trained on
    X = pd.DataFrame([row]).reindex(columns=MODEL_COLUMNS, fill_value=0)
    return X.astype(MODEL_TYPES)


from fastapi import FastAPI, HTTPException   # FastAPI builds the app; HTTPException sends error messages


# ---- 3d. Answer the request ----
# This is the vending machine: someone enters an item number,
# the app forecasts 4 weeks ahead, and sends the answer back.

HORIZON = 4   # how many weeks ahead to forecast


def forecast_product(product):
    """Forecast one product 4 weeks ahead, one week at a time.
    Each week's prediction becomes 'last week's sales' for the next week,
    the same way the notebook does it."""
    sales = list(SALES_HISTORY[product])   # a copy, so the real history isn't changed
    results = []
    for step in range(1, HORIZON + 1):
        week = LAST_WEEK + pd.Timedelta(weeks=step)       # the week we're forecasting
        X = build_features(product, week, sales)          # build its inputs (3c)
        prediction = float(np.asarray(champion_model.predict(X))[0])  # ask the model
        sales.append(prediction)                          # use it as history for next week
        results.append({"week_ending": str(week.date()),
                        "forecast_units": round(prediction)})
    return results


# Create the app (the vending machine itself)
app = FastAPI(
    title="Ceramics Weekly Demand Forecast",
    description="Enter an item number (1-50) to get its total weekly demand "
                "across all 10 stores for the next 4 weeks, from the @champion model.",
)


# The button: a request to /forecast/10 runs this for item 10
@app.get("/forecast/{item_number}")
def get_forecast(item_number: int):
    product = f"item_{item_number}"
    if product not in SALES_HISTORY:   # someone typed an item that doesn't exist
        raise HTTPException(status_code=404, detail="Item not found. Choose 1-50.")
    return {
        "product": product,
        "model": MODEL_NAME,
        "model_version": champ.version,   # shows which champion made the forecast
        "forecast": forecast_product(product),
    }

from fastapi.responses import HTMLResponse   # lets the app send a web page instead of raw data


# ---- 3e. A friendly web page ----
# The home page a distributor would use: type an item number, click Forecast,
# and see a clean table instead of raw code. Behind the scenes it calls the
# same /forecast button from 3d, so the numbers are identical.

PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Ceramics Demand Forecast</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 40px; color: #222; }
    .card { background: white; max-width: 560px; margin: auto; padding: 32px; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { margin-top: 0; font-size: 24px; }
    .sub { color: #666; margin-top: -8px; }
    input { font-size: 18px; padding: 8px; width: 90px; }
    button { font-size: 18px; padding: 8px 18px; background: #1f6feb; color: white; border: none;
             border-radius: 6px; cursor: pointer; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; }
    th, td { text-align: left; padding: 10px; border-bottom: 1px solid #ddd; }
    th { background: #f0f3f6; }
    .headline { font-size: 18px; margin-top: 20px; }
    .note { color: #666; font-size: 13px; margin-top: 16px; }
    .error { color: #c62828; margin-top: 16px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Ceramics Weekly Demand Forecast</h1>
    <p class="sub">Total units needed across all 10 stores, next 4 weeks</p>

    <label>Item number (1-50): </label>
    <input id="item" type="number" min="1" max="50" value="10">
    <button onclick="getForecast()">Forecast</button>

    <div id="result"></div>
  </div>

  <script>
    async function getForecast() {
      const item = document.getElementById("item").value;
      const box = document.getElementById("result");
      box.innerHTML = "Forecasting...";

      // Ask the app's /forecast button for this item
      const response = await fetch("/forecast/" + item);
      const data = await response.json();

      if (!response.ok) {
        box.innerHTML = '<p class="error">' + data.detail + '</p>';
        return;
      }

      // Build the results table
      const first = data.forecast[0];
      let rows = "";
      for (const week of data.forecast) {
        rows += "<tr><td>" + week.week_ending + "</td><td>"
              + week.forecast_units.toLocaleString() + "</td></tr>";
      }

      box.innerHTML =
        '<p class="headline">Order about <b>' + first.forecast_units.toLocaleString()
          + ' units</b> of item ' + item + ' for the week ending ' + first.week_ending + '.</p>'
        + "<table><tr><th>Week ending</th><th>Forecast units</th></tr>" + rows + "</table>"
        + '<p class="note">Model: ' + data.model + ' (version ' + data.model_version
          + ', current @champion)</p>';
    }
  </script>
</body>
</html>
"""


# The home page: going to http://127.0.0.1:8000 shows the page above
@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
