# Tennis Dynamics

Tennis Dynamics analyzes 2024 Wimbledon point-level data to understand how serve characteristics influence outcomes.

The project includes:
- Exploratory scripts for quick statistics (serve speed, serve location, ace rates)
- A machine learning pipeline that predicts ace probability and server point-win probability
- A Flask web app for interactive filtering and plotting of won vs lost point trends

## Project Structure

```
tennis_dynamics/
  model/
    aceserve_model.py
  Project/
    2024-wimbledon-points.csv
    acepercent.py
    servelocation.py
    speedtest.py
  website/
    app.py
    static/
      app.js
      styles.css
    templates/
      index.html
  README.md
```

## Requirements

- Python 3.10+
- pip

Python packages used in this repo:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- flask

## Setup

From the project root (`tennis_dynamics`), create and activate a virtual environment, then install dependencies.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy matplotlib seaborn scikit-learn flask
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scikit-learn flask
```

## How to Run

### 1. Run the ML pipeline

`aceserve_model.py` cleans data, engineers serve-related features, trains logistic regression models, reports accuracy metrics, and shows plots.

```powershell
cd model
python aceserve_model.py
```

### 2. Run quick analysis scripts

These scripts are in `Project/` and expect to run from that folder (they read the CSV by local filename).

```powershell
cd Project
python acepercent.py
python servelocation.py
python speedtest.py
```

### 3. Run the interactive web app

The Flask app provides filters and trend-line charts comparing won vs lost points.

```powershell
cd website
python app.py
```

Open your browser to:

`http://127.0.0.1:5000`

## What the Web App Does

- Lets you choose any numeric column for the x-axis
- Supports multiple filters (`=`, `!=`, `>`, `>=`, `<`, `<=`, `between`, `contains`, `in`)
- Plots won and lost point counts
- Automatically fits and compares trend lines (linear, logarithmic, quadratic)
- Reports adjusted $R^2$ fit scores and row counts used

## Notes

- Dataset path for the web app is resolved relative to `website/app.py` and points to `Project/2024-wimbledon-points.csv`.
- If plots do not appear when running scripts, confirm your Python environment has `matplotlib` installed and that you are running from the expected folder shown above.
