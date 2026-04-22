from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Use a non-interactive backend for server-side image rendering.
matplotlib.use("Agg")

app = Flask(__name__)

# Directory containing yearly Wimbledon CSV datasets.
PROJECT_DIR = Path(__file__).resolve().parent.parent / "Project"


def _load_data() -> pd.DataFrame:
    # Columns needed for plotting, filtering, and derived metrics.
    use_columns = [
        "P2Ace",
        "P1Ace",
        "Speed_MPH",
        "PointWinner",
        "PointServer",
        "Speed_KMH",
        "P1BreakPoint",
        "P2BreakPoint",
        "ServeIndicator",
        "Serve_Direction",
        "RallyCount",
        "ServeWidth",
        "ServeDepth",
        "ReturnDepth",
    ]

    # Load all yearly files (for example 2022, 2023, 2024).
    csv_paths = sorted(PROJECT_DIR.glob("*-wimbledon-points.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No Wimbledon points CSV files found in {PROJECT_DIR}")

    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        # Preserve the source year so users can filter by season.
        frame = pd.read_csv(path, usecols=use_columns)
        frame["DataYear"] = path.name.split("-")[0]
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)

    # Keep only valid point rows with known server/winner and real serve speed.
    df = df[(df["PointWinner"].isin([1, 2])) & (df["PointServer"].isin([1, 2])) & (df["Speed_MPH"] != 0)].copy()
    # Binary target used by outcome-based y metrics.
    df["is_won_by_server"] = (df["PointWinner"] == df["PointServer"]).astype(int)

    # Point-level flags derived from server/returner perspective.
    df["Ace"] = np.where(df["PointServer"] == 1, df["P1Ace"], df["P2Ace"])
    df["BreakPoint"] = np.where(df["PointServer"] == 1, df["P2BreakPoint"], df["P1BreakPoint"])

    # Hide player-specific ace columns from analysis UI after creating merged Ace.
    df = df.drop(columns=["P1Ace", "P2Ace"])

    return df


DF = _load_data()


def _column_meta(df: pd.DataFrame) -> list[dict[str, Any]]:
    # Metadata drives dynamic UI controls (x-axis list + filter widgets).
    meta: list[dict[str, Any]] = []
    # ServeIndicator remains filterable but hidden from x-axis selections.
    df_new = df.drop(columns=['ServeIndicator'])
    for col in df_new.columns:
        if col == "is_won_by_server":
            continue

        series = df_new[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_series.notna().mean()

        # Treat mostly numeric columns as numeric.
        if numeric_ratio > 0.9:
            clean = numeric_series.dropna()
            if clean.empty:
                continue
            meta.append(
                {
                    "name": col,
                    "type": "numeric",
                    "min": float(clean.min()),
                    "max": float(clean.max()),
                }
            )
        else:
            # Provide value lists for categorical dropdown filters.
            unique_vals = (
                series.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
            )
            unique_vals = sorted(unique_vals)
            max_dropdown_values = 200
            meta.append(
                {
                    "name": col,
                    "type": "categorical",
                    "values": unique_vals[:max_dropdown_values],
                    "values_truncated": len(unique_vals) > max_dropdown_values,
                    "total_unique": len(unique_vals),
                }
            )

    return meta


COLUMN_META = _column_meta(DF)
COLUMN_META_MAP = {item["name"]: item for item in COLUMN_META}

Y_METRICS = [
    {
        "id": "win_pct",
        "label": "Win Percentage",
        "description": "Percent chance of winning the point (wins / total points * 100).",
        "outcome_based": True,
    },
    {
        "id": "won_count",
        "label": "Won Point Count",
        "description": "Number of points won by the server at each x-value.",
        "outcome_based": True,
    },
    {
        "id": "lost_count",
        "label": "Lost Point Count",
        "description": "Number of points lost by the server at each x-value.",
        "outcome_based": True,
    },
    {
        "id": "total_points",
        "label": "Total Point Count",
        "description": "Total observed points at each x-value.",
        "outcome_based": False,
    },
]

Y_METRIC_MAP = {item["id"]: item for item in Y_METRICS}

# For these x-columns, outcome metrics are tautological and not analytically useful.
OUTCOME_INCOMPATIBLE_X_COLUMNS = {"Ace"}


def _allowed_y_metric_ids_for_x(x_column: str) -> list[str]:
    # Prevent tautological analysis (for example Ace with win percentage).
    if x_column in OUTCOME_INCOMPATIBLE_X_COLUMNS:
        return [item["id"] for item in Y_METRICS if not item["outcome_based"]]
    return [item["id"] for item in Y_METRICS]


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    # All filter rows combine with AND semantics.
    filtered = df

    for f in filters:
        column = f.get("column")
        op = f.get("op")
        value = f.get("value")

        if not column or column not in filtered.columns or op is None:
            continue

        series = filtered[column]

        # String/categorical operations.
        if op in {"eq", "neq", "contains", "in"}:
            s = series.astype(str)
            raw = "" if value is None else str(value)

            if op == "eq":
                filtered = filtered[s == raw]
            elif op == "neq":
                filtered = filtered[s != raw]
            elif op == "contains":
                filtered = filtered[s.str.contains(raw, case=False, na=False)]
            elif op == "in":
                values = [v.strip() for v in raw.split(",") if v.strip()]
                if values:
                    filtered = filtered[s.isin(values)]
            continue

        # Numeric comparisons are evaluated on coerced numeric values.
        numeric = pd.to_numeric(series, errors="coerce")
        compare_value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

        if op == "between":
            text = "" if value is None else str(value)
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) == 2:
                low = pd.to_numeric(pd.Series([parts[0]]), errors="coerce").iloc[0]
                high = pd.to_numeric(pd.Series([parts[1]]), errors="coerce").iloc[0]
                if pd.notna(low) and pd.notna(high):
                    filtered = filtered[(numeric >= low) & (numeric <= high)]
            continue

        if pd.isna(compare_value):
            continue

        if op == "gt":
            filtered = filtered[numeric > compare_value]
        elif op == "gte":
            filtered = filtered[numeric >= compare_value]
        elif op == "lt":
            filtered = filtered[numeric < compare_value]
        elif op == "lte":
            filtered = filtered[numeric <= compare_value]

    return filtered


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # R^2 goodness-of-fit metric.
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def _adjusted_r2(r2: float, n: int, params: int) -> float | None:
    # Adjusted R^2 is undefined when sample size is too small.
    if n <= params:
        return None
    return 1.0 - (1.0 - r2) * (n - 1) / (n - params)


def _json_safe_float(value: float | None) -> float | None:
    if value is None:
        return None
    val = float(value)
    if not np.isfinite(val):
        return None
    return val


def _fit_best_curve(x_vals: np.ndarray, y_vals: np.ndarray) -> dict[str, Any] | None:
    # Pick the best trend model by adjusted R^2.
    if len(x_vals) < 2:
        return None

    candidates: list[dict[str, Any]] = []

    # Linear model: y = ax + b
    linear_coef = np.polyfit(x_vals, y_vals, 1)
    linear_pred = np.polyval(linear_coef, x_vals)
    linear_r2 = _r2_score(y_vals, linear_pred)
    linear_adj_r2 = _adjusted_r2(linear_r2, len(x_vals), params=2)
    candidates.append(
        {
            "name": "linear",
            "adj_r2": linear_r2 if linear_adj_r2 is None else linear_adj_r2,
            "predict": lambda x_new, c=linear_coef: np.polyval(c, x_new),
        }
    )

    # Logarithmic model: y = a*ln(x) + b, only when x > 0.
    if np.all(x_vals > 0):
        log_x = np.log(x_vals)
        log_coef = np.polyfit(log_x, y_vals, 1)
        log_pred = log_coef[0] * log_x + log_coef[1]
        log_r2 = _r2_score(y_vals, log_pred)
        log_adj_r2 = _adjusted_r2(log_r2, len(x_vals), params=2)
        candidates.append(
            {
                "name": "logarithmic",
                "adj_r2": log_r2 if log_adj_r2 is None else log_adj_r2,
                "predict": lambda x_new, c=log_coef: c[0] * np.log(x_new) + c[1],
            }
        )

    # Quadratic model: y = ax^2 + bx + c
    if len(x_vals) >= 3:
        quad_coef = np.polyfit(x_vals, y_vals, 2)
        quad_pred = np.polyval(quad_coef, x_vals)
        quad_r2 = _r2_score(y_vals, quad_pred)
        quad_adj_r2 = _adjusted_r2(quad_r2, len(x_vals), params=3)
        candidates.append(
            {
                "name": "quadratic",
                "adj_r2": quad_r2 if quad_adj_r2 is None else quad_adj_r2,
                "predict": lambda x_new, c=quad_coef: np.polyval(c, x_new),
            }
        )

    if not candidates:
        return None

    best = max(candidates, key=lambda item: item["adj_r2"])
    x_line = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), 200)

    # Ensure log predictions only receive positive x values.
    if best["name"] == "logarithmic":
        x_line = x_line[x_line > 0]
        if len(x_line) == 0:
            return None

    y_line = best["predict"](x_line)
    return {
        "model": best["name"],
        "score": float(best["adj_r2"]),
        "x_line": x_line,
        "y_line": y_line,
    }


def _build_plot(df: pd.DataFrame, x_column: str, y_metric: str) -> tuple[str, dict[str, Any]]:
    # Convert selected x column to numeric coordinates for plotting.
    x = pd.to_numeric(df[x_column], errors="coerce")
    work = df.copy()
    work["x"] = x
    work = work.dropna(subset=["x"]) 

    if work.empty:
        raise ValueError("No rows remain after applying filters for the selected X-axis.")

    # Bin dense x values to keep charts readable and stable.
    unique_count = work["x"].nunique()

    if unique_count > 80:
        bins = np.linspace(work["x"].min(), work["x"].max(), 31)
        work["x_bucket"] = pd.cut(work["x"], bins=bins, include_lowest=True)
        grouped = work.groupby("x_bucket", observed=True)["is_won_by_server"].agg(["sum", "count"]).reset_index()
        grouped["x_value"] = grouped["x_bucket"].apply(lambda interval: interval.mid if pd.notna(interval) else np.nan)
    else:
        grouped = work.groupby("x", observed=True)["is_won_by_server"].agg(["sum", "count"]).reset_index()
        grouped = grouped.rename(columns={"x": "x_value"})

    grouped["won_count"] = grouped["sum"]
    grouped["lost_count"] = grouped["count"] - grouped["sum"]
    grouped["total_points"] = grouped["count"]
    # Default analytic metric: empirical win probability at each x-value.
    grouped["win_pct"] = np.where(grouped["total_points"] > 0, (grouped["won_count"] / grouped["total_points"]) * 100.0, np.nan)
    grouped = grouped.dropna(subset=["x_value"]).sort_values("x_value")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_vals = grouped["x_value"].to_numpy(dtype=float)

    # Y metric is selected by UI and validated against supported aggregates.
    if y_metric not in grouped.columns:
        raise ValueError(f"Unsupported y_metric: {y_metric}")

    y_vals = grouped[y_metric].to_numpy(dtype=float)
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]

    if len(x_vals) == 0:
        raise ValueError("No valid points remain for the selected x/y configuration.")

    y_label = Y_METRIC_MAP[y_metric]["label"]
    y_model = None
    y_score = None

    ax.scatter(x_vals, y_vals, alpha=0.78, color="#1f5fbf", label=y_label)

    # Overlay adaptive best-fit curve (linear / logarithmic / quadratic).
    if len(x_vals) >= 2:
        y_fit = _fit_best_curve(x_vals, y_vals)
        if y_fit is not None:
            y_model = y_fit["model"]
            y_score = y_fit["score"]
            ax.plot(
                y_fit["x_line"],
                y_fit["y_line"],
                color="#113a75",
                linewidth=2,
                label=f"Best-fit ({y_model})",
            )

    ax.set_title(f"{y_label} by {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)

    # Return a base64 image so frontend can display without file writes.
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    summary = {
        "rows_used": int(len(work)),
        "won_points": int(work["is_won_by_server"].sum()),
        "lost_points": int((1 - work["is_won_by_server"]).sum()),
        "overall_win_pct": _json_safe_float(work["is_won_by_server"].mean() * 100.0),
        "selected_y_metric": y_metric,
        "selected_y_label": y_label,
        "trend_model": y_model,
        "fit_score": _json_safe_float(y_score),
    }

    return encoded, summary


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/meta")
def get_meta():
    # Frontend bootstrap endpoint (columns, y metrics, defaults, compatibility).
    numeric_columns = [m["name"] for m in COLUMN_META if m["type"] == "numeric" and m["name"] != "ServeIndicator"]
    y_metrics_with_compatibility = []
    for metric in Y_METRICS:
        compatible_x = [col for col in numeric_columns if metric["id"] in _allowed_y_metric_ids_for_x(col)]
        y_metrics_with_compatibility.append(
            {
                "id": metric["id"],
                "label": metric["label"],
                "description": metric["description"],
                "compatible_x": compatible_x,
            }
        )

    return jsonify({
        "columns": COLUMN_META,
        "default_x": "Speed_MPH" if "Speed_MPH" in numeric_columns else numeric_columns[0],
        "y_metrics": y_metrics_with_compatibility,
        "default_y": "win_pct",
    })


@app.post("/api/plot")
def plot_data():
    # Render endpoint for filtered x/y requests from the UI.
    payload = request.get_json(silent=True) or {}

    x_column = payload.get("x_column")
    y_metric = payload.get("y_metric", "win_pct")
    filters = payload.get("filters", [])

    if not x_column or x_column not in DF.columns:
        return jsonify({"error": "Invalid x_column"}), 400
    if x_column == "ServeIndicator":
        return jsonify({"error": "ServeIndicator cannot be used as an x-axis"}), 400

    if COLUMN_META_MAP.get(x_column, {}).get("type") != "numeric":
        return jsonify({"error": "x_column must be numeric"}), 400

    if y_metric not in Y_METRIC_MAP:
        return jsonify({"error": "Invalid y_metric"}), 400

    allowed_metrics = _allowed_y_metric_ids_for_x(x_column)
    if y_metric not in allowed_metrics:
        return jsonify(
            {
                "error": f"{Y_METRIC_MAP[y_metric]['label']} is not valid when x-axis is {x_column}.",
                "allowed_y_metrics": allowed_metrics,
            }
        ), 400

    filtered = _apply_filters(DF, filters)

    if filtered.empty:
        return jsonify({"error": "No rows match the selected filters."}), 400

    try:
        image_b64, summary = _build_plot(filtered, x_column, y_metric)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "image": image_b64,
        "summary": summary,
    })


if __name__ == "__main__":
    app.run(debug=True)
