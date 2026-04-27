# Tennis Dynamics

Tennis Dynamics analyzes Wimbledon point-level data (2022, 2023, 2024) to understand how serve and rally characteristics influence outcomes.

The project includes:
- Exploratory scripts for quick statistics (serve speed, serve location, ace rates)
- A machine learning pipeline that predicts ace probability and server point-win probability
- A Flask web app for interactive filtering and plotting of won vs lost point trends
- Link to our presentation

## Project Presentation Link

- https://youtu.be/9nO38lWX0IY

## Project Structure

```
tennis_dynamics/
  model/
    aceserve_model.py
  Project/
    2022-wimbledon-points.csv
    2023-wimbledon-points.csv
    2024-wimbledon-points.csv
    acepercent.py
    matchSpeedChanges.py
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

The Flask app provides interactive x/y analysis, filters, and adaptive trend-line charts.

```powershell
cd website
python app.py
```

Open your browser to:

`http://127.0.0.1:5000`

## Website Walkthrough (Team-Friendly)

This section explains the web app in plain language so everyone can use and present results consistently.

### 1. What data is loaded

- The app automatically combines all yearly files in `Project/` that match `*-wimbledon-points.csv`.
- Right now that means 2022, 2023, and 2024 are all included.
- The app adds a `DataYear` column so you can filter by season.

### 2. How to use the controls

- **X-axis category:** the factor you want to test (for example `Speed_MPH`, `RallyCount`, `ServeWidth`).
- **Y-axis metric:** what you want to measure against that factor.
  - Default is **Win Percentage** (wins / total points * 100), which is the main analytic metric.
  - Other options include won count, lost count, and total points.
- **Recommended badge:** points to the most useful y-axis metric for the chosen x-axis.

### 3. Filters (how they combine)

- You can add multiple filter rows.
- Filters are combined with **AND** logic.
  - Example: `DataYear = 2024` **AND** `ServeWidth = C` keeps only rows matching both.

Available operators:

- Numeric-friendly: `=`, `!=`, `>`, `>=`, `<`, `<=`, `between`
- String-friendly: `=`, `!=`, `contains`, `in`

Examples:

- `between` expects `min,max` such as `100,130`
- `in` expects comma-separated values such as `C,BC,BW`

### 4. Why some x/y combinations are blocked

- The app intentionally blocks combinations that are tautological (not meaningful), such as using `Ace` to predict win percentage.
- This prevents misleading charts and keeps the analysis focused on useful relationships.

### 5. How to read the chart

- Dots are observed grouped results for each x-value (or x-bin for dense numeric ranges).
- The line is an adaptive best-fit trend.
  - The app tests linear, logarithmic, and quadratic models.
  - It selects the best one using adjusted $R^2$.
- Summary cards below the chart report:
  - Rows used after filters
  - Won/lost totals
  - Overall win chance
  - Selected y metric
  - Trend model and adjusted $R^2$

## Notes

- Dataset paths for the web app are resolved relative to `website/app.py` and loaded from all matching CSV files in `Project/`.
- If plots do not appear when running scripts, confirm your Python environment has `matplotlib` installed and that you are running from the expected folder shown above.
