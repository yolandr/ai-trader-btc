"""
diagnose_model.py
Melihat distribusi prediksi & confidence model pada data historis,
untuk membantu menentukan CONFIDENCE_THRESHOLD yang wajar.

Cara pakai:
    python diagnose_model.py
"""

import numpy as np
import joblib
import tensorflow as tf

import config
import data_fetcher
from indicators import add_technical_indicators
from dataset import FEATURE_COLUMNS, label_trend


def main():
    print("Mengambil data & memuat model ...")
    raw_df = data_fetcher.fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, 2000)
    df = add_technical_indicators(raw_df)
    df["label"] = label_trend(df)
    df.dropna(inplace=True)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)

    features = df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)
    window = config.LOOKBACK_WINDOW

    all_confidences = []
    all_preds = []

    for i in range(window, len(features_scaled)):
        X = np.expand_dims(features_scaled[i - window:i], axis=0)
        probs = model.predict(X, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        all_confidences.append(confidence)
        all_preds.append(pred_class)

    all_confidences = np.array(all_confidences)
    all_preds = np.array(all_preds)

    print("\n===== DISTRIBUSI PREDIKSI =====")
    print(f"Total prediksi: {len(all_preds)}")
    print(f"DOWN (0)     : {(all_preds == 0).sum()} ({(all_preds == 0).mean():.1%})")
    print(f"SIDEWAYS (1) : {(all_preds == 1).sum()} ({(all_preds == 1).mean():.1%})")
    print(f"UP (2)        : {(all_preds == 2).sum()} ({(all_preds == 2).mean():.1%})")

    print("\n===== DISTRIBUSI CONFIDENCE =====")
    print(f"Min      : {all_confidences.min():.3f}")
    print(f"Max      : {all_confidences.max():.3f}")
    print(f"Rata-rata : {all_confidences.mean():.3f}")
    print(f"Median   : {np.median(all_confidences):.3f}")

    for pct in [50, 60, 70, 80, 90]:
        threshold = np.percentile(all_confidences, pct)
        print(f"Persentil {pct}%: confidence >= {threshold:.3f}")

    print("\n===== SARAN THRESHOLD =====")
    for t in [0.35, 0.40, 0.45, 0.50, 0.55]:
        n_qualified = (all_confidences >= t).sum()
        pct = n_qualified / len(all_confidences) * 100
        print(f"Threshold {t}: {n_qualified} prediksi lolos ({pct:.1f}% dari total)")


if __name__ == "__main__":
    main()