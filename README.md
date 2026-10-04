# Ceramics Demand Forecasting

## Description
This project develops a machine learning workflow to forecast daily demand for ceramics products. Using historical sales data, the team will prepare store- and item-level data, explore sales patterns, build forecasting models, and evaluate forecast performance.
The project is being developed in Python using Google Colab. The current notebook imports the training and testing data, saves unchanged copies of the files, and loads the datasets into pandas DataFrames for analysis.

## Project Goal
The goal of this project is to create a reproducible demand forecasting process that can help estimate future product sales based on historical daily sales records.

The team plans to:

- Load and organize the raw training and testing datasets
- Preserve unchanged copies of the original files
- Explore patterns in sales by date, store, and item
- Prepare data for forecasting
- Build and evaluate demand forecasting models
- Generate forecasts for the test dataset

## Dataset
The project uses two CSV files:

- `train.csv` — Historical sales data used to train forecasting models
- `test.csv` — Data used to generate future sales forecasts

The training data includes the following variables:

| Variable | Description |
|---|---|
| `date` | Date of the sales observation |
| `store` | Store identification number |
| `item` | Item identification number |
| `sales` | Number of units sold |

The test data includes:

| Variable | Description |
|---|---|
| `date` | Date for the forecast |
| `store` | Store identification number |
| `item` | Item identification number |

## Project Structure
```text
ceramics-demand-forecasting/
│
├── MVPWeek5.ipynb
├── README.md
```

- `MVPWeek5.ipynb` contains the project workflow and analysis code.
- `raw_data/` stores unchanged copies of the original datasets.

## Requirements
This project is designed to run in Google Colab with Python. The notebook currently uses the following Python libraries:

```python
import pandas as pd
import os
import shutil
from google.colab import files
```

## Installation and Setup
1. Clone or download this repository.
2. Open `MVPWeek5.ipynb` in Google Colab.
3. Run the first code cell to import the required libraries.
4. Upload `train.csv` and `test.csv` when prompted.
5. Run the remaining notebook cells in order.

## Usage
The following code uploads the training and testing files in Google Colab:

```python
from google.colab import files

uploaded = files.upload()
```

The notebook then saves unchanged copies of the uploaded files in the `raw_data` folder:

```python
import os
import shutil

os.makedirs("raw_data", exist_ok=True)

shutil.copy("train.csv", "raw_data/train.csv")
shutil.copy("test.csv", "raw_data/test.csv")
```

Next, the data can be loaded into pandas DataFrames:

```python
import pandas as pd

train_df = pd.read_csv("raw_data/train.csv")
test_df = pd.read_csv("raw_data/test.csv")

print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
```

## Collaboration Workflow
The team uses GitHub to manage project files and collaborate on changes.
- Use GitHub Issues to document bugs, questions, and proposed improvements.
- Create a separate branch before making a change.
- Open a pull request when the change is ready for review.
- Link the pull request to its related issue using `Closes #issue-number`.
- Have another team member review the pull request before merging it into the `main` branch.

## Current Status
The project is currently in the data-loading and data-preparation stage.

Completed tasks:
- Created the GitHub repository
- Added collaborators to the repository
- Uploaded the initial Google Colab notebook
- Imported the training and testing datasets
- Created folders for raw data, processed data, and forecasts
- Saved unchanged copies of the training and testing data

Planned next steps:
- Improve file-upload handling when Colab changes uploaded filenames
- Clean and validate the datasets
- Conduct exploratory data analysis
- Engineer forecasting features
- Build demand forecasting models
- Evaluate model performance
- Generate final sales forecasts

## Contributors
- Anushka Maharana
- Caroline Hamilton
- Marshawn Amison
- Michelle Hudson
- Paolo Lujan

