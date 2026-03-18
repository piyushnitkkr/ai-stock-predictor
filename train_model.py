import numpy as np
import joblib
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from data_fetcher import get_stock_data, get_nifty_index
from features import add_indicators
from model_transformer import build_model
from model_xgb import build_xgb
from nifty50 import nifty50


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SEQUENCE_LENGTH = 120
LABEL_THRESHOLD = 0.015
TOP_FEATURES = 10  # Select top 10 features only

# ── Step 1: Collect all training data ─────────────────────────────────────────
nifty_index = get_nifty_index()

all_features = []
all_closes   = []
feature_names = None

for ticker in nifty50:
    logging.info("Downloading %s", ticker)
    data = get_stock_data(ticker)

    if data is None or data.empty:
        logging.warning("Skipping %s — no data", ticker)
        continue

    try:
        data = add_indicators(data, nifty_index)
    except Exception as e:
        logging.warning("Indicators failed for %s: %s", ticker, e)
        continue

    if len(data) < SEQUENCE_LENGTH + 2:
        logging.warning("Skipping %s — too few rows (%d)", ticker, len(data))
        continue

    # Get Close column reliably
    close_col = next((c for c in data.columns if "Close" in str(c)), None)
    if close_col is None:
        logging.warning("Skipping %s — no Close column", ticker)
        continue

    # Store feature names from first stock
    if feature_names is None:
        feature_names = [str(col) for col in data.columns]
        logging.info(f"Feature names: {feature_names}")

    close    = data[close_col].squeeze().values.astype(float)
    features = data.values.astype(float)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    all_features.append(features)
    all_closes.append(close)

if not all_features:
    raise RuntimeError("No training data collected.")

# ── Step 2: Build sequences for feature selection ────────────────────────────
logging.info("Building sequences for feature selection...")

X_flat = []  # For feature selection (flattened)
X_seq = []   # For transformer (sequences)
y_all = []

for features, close in zip(all_features, all_closes):
    for i in range(SEQUENCE_LENGTH, len(features) - 1):
        # Current features (last row of sequence)
        current_features = features[i]
        X_flat.append(current_features)

        # Full sequence for transformer
        sequence = features[i - SEQUENCE_LENGTH : i]
        X_seq.append(sequence)

        # Label from next day's return
        pct = (close[i + 1] - close[i]) / (close[i] + 1e-9)
        if pct > LABEL_THRESHOLD:
            y_all.append(0)   # bullish
        elif pct < -LABEL_THRESHOLD:
            y_all.append(1)   # bearish
        else:
            y_all.append(2)   # sideways

X_flat = np.array(X_flat, dtype=np.float32)
X_seq = np.array(X_seq, dtype=np.float32)
y = np.array(y_all, dtype=np.int32)

logging.info(f"Data shape: X_flat={X_flat.shape}, X_seq={X_seq.shape}, y={y.shape}")
logging.info(f"Class distribution: Bullish={np.sum(y==0)}, Bearish={np.sum(y==1)}, Sideways={np.sum(y==2)}")

# ── Step 3: FEATURE SELECTION ─────────────────────────────────────────────────
logging.info("=== FEATURE SELECTION ===")

# Method 1: Mutual Information (non-linear relationships)
mi_scores = mutual_info_classif(X_flat, y, random_state=42)
mi_ranking = np.argsort(mi_scores)[::-1]

# Method 2: F-statistic (linear relationships)
f_scores, _ = f_classif(X_flat, y)
f_ranking = np.argsort(f_scores)[::-1]

# Method 3: Random Forest feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_flat, y)
rf_importance = rf.feature_importances_
rf_ranking = np.argsort(rf_importance)[::-1]

# Combine rankings (ensemble approach)
feature_scores = {}
for i, feat_name in enumerate(feature_names):
    # Weighted score: MI (40%) + F-stat (30%) + RF (30%)
    mi_rank = np.where(mi_ranking == i)[0][0]
    f_rank = np.where(f_ranking == i)[0][0]
    rf_rank = np.where(rf_ranking == i)[0][0]

    combined_score = 0.4 * (len(feature_names) - mi_rank) + \
                     0.3 * (len(feature_names) - f_rank) + \
                     0.3 * (len(feature_names) - rf_rank)
    feature_scores[feat_name] = combined_score

# Select top features
top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
selected_features = [name for name, score in top_features[:TOP_FEATURES]]
selected_indices = [feature_names.index(name) for name in selected_features]

logging.info("=== TOP FEATURES SELECTED ===")
for i, (name, score) in enumerate(top_features[:TOP_FEATURES]):
    logging.info(f"{i+1:2d}. {name:15s} (score: {score:.1f})")

# ── Step 4: Apply feature selection ───────────────────────────────────────────
X_flat_selected = X_flat[:, selected_indices]
X_seq_selected = X_seq[:, :, selected_indices]

# Fit scaler on selected features only
scaler = StandardScaler()
scaler.fit(X_flat_selected)

# Scale the sequence data
X_scaled_sequences = []
for seq in X_seq_selected:
    scaled_seq = scaler.transform(seq)
    X_scaled_sequences.append(scaled_seq)
X_scaled = np.array(X_scaled_sequences, dtype=np.float32)

logging.info(f"Selected feature shape: {X_scaled.shape}")

# ── Step 5: Train/Val split ───────────────────────────────────────────────────
split_idx = int(len(X_scaled) * 0.75)
older_data = X_scaled[:split_idx]
older_labels = y[:split_idx]
recent_data = X_scaled[split_idx:]
recent_labels = y[split_idx:]

X_train, X_val, y_train_int, y_val_int = train_test_split(
    older_data, older_labels,
    test_size=0.25,
    random_state=42,
    stratify=older_labels
)

# Add recent data
recent_split = int(len(recent_data) * 0.7)
X_train = np.vstack([X_train, recent_data[:recent_split]])
y_train_int = np.hstack([y_train_int, recent_labels[:recent_split]])
X_val = np.vstack([X_val, recent_data[recent_split:]])
y_val_int = np.hstack([y_val_int, recent_labels[recent_split:]])

y_train = to_categorical(y_train_int, 3)
y_val = to_categorical(y_val_int, 3)

logging.info(f"Final training: X_train={X_train.shape}, X_val={X_val.shape}")

# ── Step 6: Train Transformer ─────────────────────────────────────────────────
callbacks = [EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True, verbose=1)]

model = build_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=callbacks)
model.save("transformer_model.keras")

# ── Step 7: Train XGBoost ─────────────────────────────────────────────────────
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

xgb_model = build_xgb()
xgb_model.fit(X_train_flat, y_train_int, eval_set=[(X_val_flat, y_val_int)], verbose=100)

# ── Step 8: Save everything ───────────────────────────────────────────────────
joblib.dump(xgb_model, "xgb_model.save")
joblib.dump(scaler, "scaler.save")

# Save feature selection info
feature_info = {
    'selected_features': selected_features,
    'selected_indices': selected_indices,
    'feature_names': feature_names,
    'feature_scores': feature_scores
}
joblib.dump(feature_info, "feature_selection.save")

logging.info("=== TRAINING COMPLETE ===")
logging.info(f"Selected {TOP_FEATURES} best features out of {len(feature_names)}")
logging.info("Saved: transformer_model.keras, xgb_model.save, scaler.save, feature_selection.save")