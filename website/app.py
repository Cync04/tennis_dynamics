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

DATA_PATH = Path(__file__).resolve().parent.parent / "Project" / "2024-wimbledon-points.csv"


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Keep only rows that represent actual served points with known winner/server.
    df = df[(df["PointWinner"].isin([1, 2])) & (df["PointServer"].isin([1, 2])) & (df["Speed_MPH"] != (0))].copy()
    df["is_won_by_server"] = (df["PointWinner"] == df["PointServer"]).astype(int)

    return df


DF = _load_data()


def _column_meta(df: pd.DataFrame) -> list[dict[str, Any]]:
    meta: list[dict[str, Any]] = []

    for col in df.columns:
        if col == "is_won_by_server":
            continue

        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_series.notna().mean()

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
            unique_vals = (
                series.dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
            )
            unique_vals = sorted(unique_vals)
            meta.append(
                {
                    "name": col,
                    "type": "categorical",
                    "values": unique_vals[:40],
                    "total_unique": len(unique_vals),
                }
            )

    return meta


COLUMN_META = _column_meta(DF)
COLUMN_META_MAP = {item["name"]: item for item in COLUMN_META}


def _apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    filtered = df

    for f in filters:
        column = f.get("column")
        op = f.get("op")
        value = f.get("value")

        if not column or column not in filtered.columns or op is None:
            continue

        series = filtered[column]

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
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - (ss_res / ss_tot)


def _adjusted_r2(r2: float, n: int, params: int) -> float:
    # Penalize extra parameters so quadratic does not win by default.
    if n <= params:
        return float("-inf")
    return 1.0 - (1.0 - r2) * (n - 1) / (n - params)


def _fit_best_curve(x_vals: np.ndarray, y_vals: np.ndarray) -> dict[str, Any] | None:
    if len(x_vals) < 2:
        return None

    candidates: list[dict[str, Any]] = []

    # Linear model: y = ax + b
    linear_coef = np.polyfit(x_vals, y_vals, 1)
    linear_pred = np.polyval(linear_coef, x_vals)
    linear_adj_r2 = _adjusted_r2(_r2_score(y_vals, linear_pred), len(x_vals), params=2)
    candidates.append(
        {
            "name": "linear",
            "adj_r2": linear_adj_r2,
            "predict": lambda x_new, c=linear_coef: np.polyval(c, x_new),
        }
    )

    # Logarithmic model: y = a*ln(x) + b, only when x > 0.
    if np.all(x_vals > 0):
        log_x = np.log(x_vals)
        log_coef = np.polyfit(log_x, y_vals, 1)
        log_pred = log_coef[0] * log_x + log_coef[1]
        log_adj_r2 = _adjusted_r2(_r2_score(y_vals, log_pred), len(x_vals), params=2)
        candidates.append(
            {
                "name": "logarithmic",
                "adj_r2": log_adj_r2,
                "predict": lambda x_new, c=log_coef: c[0] * np.log(x_new) + c[1],
            }
        )

    # Quadratic model: y = ax^2 + bx + c
    if len(x_vals) >= 3:
        quad_coef = np.polyfit(x_vals, y_vals, 2)
        quad_pred = np.polyval(quad_coef, x_vals)
        quad_adj_r2 = _adjusted_r2(_r2_score(y_vals, quad_pred), len(x_vals), params=3)
        candidates.append(
            {
                "name": "quadratic",
                "adj_r2": quad_adj_r2,
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


def _build_plot(df: pd.DataFrame, x_column: str) -> tuple[str, dict[str, Any]]:
    x = pd.to_numeric(df[x_column], errors="coerce")
    work = df.copy()
    work["x"] = x
    work = work.dropna(subset=["x"]) 

    if work.empty:
        raise ValueError("No rows remain after applying filters for the selected X-axis.")

    # Use bins for very high-cardinality numeric columns to keep the chart readable.
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
    grouped = grouped.dropna(subset=["x_value"]).sort_values("x_value")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_vals = grouped["x_value"].to_numpy(dtype=float)
    won_vals = grouped["won_count"].to_numpy(dtype=float)
    lost_vals = grouped["lost_count"].to_numpy(dtype=float)

    ax.scatter(x_vals, won_vals, alpha=0.75, color="#1b8f4b", label="Won points")
    ax.scatter(x_vals, lost_vals, alpha=0.75, color="#cc3d3d", label="Lost points")

    won_model = None
    lost_model = None
    won_score = None
    lost_score = None

    if len(x_vals) >= 2:
        won_fit = _fit_best_curve(x_vals, won_vals)
        if won_fit is not None:
            won_model = won_fit["model"]
            won_score = won_fit["score"]
            ax.plot(
                won_fit["x_line"],
                won_fit["y_line"],
                color="#0c5f2e",
                linewidth=2,
                label=f"Won best-fit ({won_model})",
            )

        lost_fit = _fit_best_curve(x_vals, lost_vals)
        if lost_fit is not None:
            lost_model = lost_fit["model"]
            lost_score = lost_fit["score"]
            ax.plot(
                lost_fit["x_line"],
                lost_fit["y_line"],
                color="#8f1f1f",
                linewidth=2,
                label=f"Lost best-fit ({lost_model})",
            )

    ax.set_title(f"Point outcomes by {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel("Point count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    summary = {
        "rows_used": int(len(work)),
        "won_points": int(work["is_won_by_server"].sum()),
        "lost_points": int((1 - work["is_won_by_server"]).sum()),
        "won_trend_model": won_model,
        "lost_trend_model": lost_model,
        "won_fit_score": won_score,
        "lost_fit_score": lost_score,
    }

    return encoded, summary


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/meta")
def get_meta():
    numeric_columns = [m["name"] for m in COLUMN_META if m["type"] == "numeric" and m["name"] != "PointWinner"]
    return jsonify({
        "columns": COLUMN_META,
        "default_x": "Speed_MPH" if "Speed_MPH" in numeric_columns else numeric_columns[0],
    })


@app.post("/api/plot")
def plot_data():
    payload = request.get_json(silent=True) or {}

    x_column = payload.get("x_column")
    filters = payload.get("filters", [])

    if not x_column or x_column not in DF.columns:
        return jsonify({"error": "Invalid x_column"}), 400

    if COLUMN_META_MAP.get(x_column, {}).get("type") != "numeric":
        return jsonify({"error": "x_column must be numeric"}), 400

    filtered = _apply_filters(DF, filters)

    if filtered.empty:
        return jsonify({"error": "No rows match the selected filters."}), 400

    try:
        image_b64, summary = _build_plot(filtered, x_column)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "image": image_b64,
        "summary": summary,
    })


if __name__ == "__main__":
    app.run(debug=True)
