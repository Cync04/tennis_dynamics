import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------
# STEP 1 — LOAD & CLEAN DATASET
# ---------------------------------------------------------
df = pd.read_csv("../Project/2024-wimbledon-points.csv")
print("Initial rows:", len(df))

# Keep rows where Speed_KMH exists (but allow 0)
df = df[df["Speed_KMH"].notna()]

# Keep realistic speeds but allow 0 for missing
df = df[(df["Speed_KMH"] <= 240)]

# Clean ServeNumber safely
df["ServeNumber"] = pd.to_numeric(df["ServeNumber"], errors="coerce")
df = df[df["ServeNumber"].notna()]
df["ServeNumber"] = df["ServeNumber"].astype(int)

# Clean ServeWidth (contains direction)
df["ServeWidth"] = df["ServeWidth"].astype(str).str.strip().str.upper()
df = df[df["ServeWidth"].isin(["C", "BC", "B", "BW", "W"])]

# Reset index after cleaning
df = df.reset_index(drop=True)

# ---------------------------------------------------------
# STEP 2 — SERVER PERSPECTIVE COLUMNS
# ---------------------------------------------------------

# Server ace
df["server_ace"] = df.apply(
    lambda row: row["P1Ace"] if row["PointServer"] == 1 else row["P2Ace"],
    axis=1
)

# Server point won
df["server_point_won"] = df.apply(
    lambda row: 1 if row["PointWinner"] == row["PointServer"] else 0,
    axis=1
)

# Server double fault
df["server_double_fault"] = df.apply(
    lambda row: row["P1DoubleFault"] if row["PointServer"] == 1 else row["P2DoubleFault"],
    axis=1
)

# Server first serve in
df["server_first_serve_in"] = df.apply(
    lambda row: row["P1FirstSrvIn"] if row["PointServer"] == 1 else row["P2FirstSrvIn"],
    axis=1
)

# Ensure server_point_won is numeric
df["server_point_won"] = df["server_point_won"].astype(int)

# Remove rows where target variables are missing
df = df[df["server_ace"].notna()]
df = df[df["server_point_won"].notna()]

# Final reset
df = df.reset_index(drop=True)
print("Final rows after cleaning + server perspective:", len(df))

# ---------------------------------------------------------
# STEP 3 — FEATURE ENGINEERING
# ---------------------------------------------------------

# One-hot encode ServeWidth (C, BC, B, BW, W)
width_dummies = pd.get_dummies(df["ServeWidth"], prefix="width")
df = pd.concat([df, width_dummies], axis=1)

# Encode serve number
df["is_first_serve"] = (df["ServeNumber"] == 1).astype(int)
df["is_second_serve"] = (df["ServeNumber"] == 2).astype(int)

# Break point pressure indicator
df["server_break_point_against"] = df.apply(
    lambda row: row["P2BreakPoint"] if row["PointServer"] == 1 else row["P1BreakPoint"],
    axis=1
).astype(int)

# Normalize serve speed
df["speed_norm"] = (df["Speed_KMH"] - df["Speed_KMH"].mean()) / df["Speed_KMH"].std()

# Rally length feature
df["rally_len"] = df["RallyCount"].fillna(0)

# Feature matrix for ace model
ace_features = [
    "speed_norm",
    "is_first_serve",
    "is_second_serve",
    "server_break_point_against",
] + list(width_dummies.columns)

X_ace = df[ace_features]
y_ace = df["server_ace"]

# Feature matrix for point-win model
point_features = ace_features + ["rally_len"]

X_point = df[point_features]
y_point = df["server_point_won"]

# ---------------------------------------------------------
# STEP 4 — LOGISTIC REGRESSION MODELS
# ---------------------------------------------------------

# ---------------------------------------------------------
# ACE MODEL
# ---------------------------------------------------------

# Train/test split for ace model
X_train_ace, X_test_ace, y_train_ace, y_test_ace = train_test_split(
    X_ace, y_ace, test_size=0.2, random_state=42
)

# Train logistic regression (ace model)
ace_model = LogisticRegression(max_iter=1000)
ace_model.fit(X_train_ace, y_train_ace)

# Evaluate ace model (single split)
ace_preds = ace_model.predict(X_test_ace)
ace_acc = accuracy_score(y_test_ace, ace_preds)
print("Ace model accuracy (train/test split):", ace_acc)

# Cross-validation accuracy
ace_cv_scores = cross_val_score(
    ace_model, X_ace, y_ace, cv=5, scoring="accuracy"
)
print("Ace model CV accuracy:", ace_cv_scores.mean())
print("Ace model CV fold scores:", ace_cv_scores)

# Show ace model coefficients
print("\nAce model coefficients:")
for feature, coef in zip(X_ace.columns, ace_model.coef_[0]):
    print(f"{feature}: {coef:.4f}")


# ---------------------------------------------------------
# POINT-WIN MODEL
# ---------------------------------------------------------

# Train/test split for point‑win model
X_train_point, X_test_point, y_train_point, y_test_point = train_test_split(
    X_point, y_point, test_size=0.2, random_state=42
)

# Train logistic regression (point‑win model)
point_model = LogisticRegression(max_iter=1000)
point_model.fit(X_train_point, y_train_point)

# Evaluate point‑win model (single split)
point_preds = point_model.predict(X_test_point)
point_acc = accuracy_score(y_test_point, point_preds)
print("\nPoint‑win model accuracy (train/test split):", point_acc)

# Cross-validation accuracy
point_cv_scores = cross_val_score(
    point_model, X_point, y_point, cv=5, scoring="accuracy"
)
print("Point‑win model CV accuracy:", point_cv_scores.mean())
print("Point‑win model CV fold scores:", point_cv_scores)

# Show point‑win model coefficients
print("\nPoint‑win model coefficients:")
for feature, coef in zip(X_point.columns, point_model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

# ---------------------------------------------------------
# STEP 5 — VISUALIZATIONS
# ---------------------------------------------------------

# =========================================================
# 1. COEFFICIENT BAR CHART — POINT‑WIN MODEL
# =========================================================
coef_values = point_model.coef_[0]
features = X_point.columns

plt.figure(figsize=(10, 5))
plt.bar(features, coef_values)
plt.xticks(rotation=45, ha='right')
plt.title("Point‑Win Model Coefficients")
plt.ylabel("Coefficient Value")
plt.xlabel("Metrics")
plt.tight_layout()
plt.show()

# =========================================================
# 2. POINT‑WIN PROBABILITY VS SERVE SPEED
# =========================================================
speed_range = np.linspace(df["Speed_KMH"].min(), df["Speed_KMH"].max(), 200)
speed_norm_range = (speed_range - df["Speed_KMH"].mean()) / df["Speed_KMH"].std()

# Build temp DataFrame with ALL features used during training
temp = pd.DataFrame({
    "speed_norm": speed_norm_range,
    "is_first_serve": 1,
    "is_second_serve": 0,
    "server_break_point_against": 0,
    "rally_len": df["rally_len"].mean(),
})

# Add ALL width dummy columns (neutral)
for col in width_dummies.columns:
    temp[col] = 0

# Reorder columns to EXACTLY match training order
temp = temp[point_features]

probs = point_model.predict_proba(temp)[:, 1]

plt.figure(figsize=(8, 5))
plt.plot(speed_range, probs)
plt.xlabel("Serve Speed (km/h)")
plt.ylabel("Predicted Point‑Win Probability")
plt.title("Point‑Win Probability vs Serve Speed (First Serve)")
plt.grid(True)
plt.show()

# =========================================================
# 3. ACE PROBABILITY VS SPEED — BY SERVE WIDTH
# =========================================================
plt.figure(figsize=(8, 5))

widths = list(width_dummies.columns)
colors = ["blue", "green", "red", "purple", "orange"]

for width, color in zip(widths, colors):

    # Build temp DataFrame with ALL features used during training
    temp_w = pd.DataFrame({
        "speed_norm": speed_norm_range,
        "is_first_serve": 1,
        "is_second_serve": 0,
        "server_break_point_against": 0,
    })

    # Add ALL width dummy columns (in correct order)
    for col in widths:
        temp_w[col] = 0

    # Activate the current width
    temp_w[width] = 1

    # Reorder columns to EXACTLY match training order
    temp_w = temp_w[ace_features]

    ace_probs_w = ace_model.predict_proba(temp_w)[:, 1]
    plt.plot(speed_range, ace_probs_w, label=width.replace("width_", ""), color=color)

plt.xlabel("Serve Speed (km/h)")
plt.ylabel("Predicted Ace Probability")
plt.title("Ace Probability vs Serve Speed by Serve Width (First Serve)")
plt.legend(title="Width")
plt.grid(True)
plt.show()