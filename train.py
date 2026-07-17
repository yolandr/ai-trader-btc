"""
train.py
Script utama untuk melatih model prediksi tren BTC.

Cara pakai:
    python train.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa GUI display
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import config
import data_fetcher
from dataset import prepare_training_data
from model import build_model

CLASS_NAMES = ["DOWN", "SIDEWAYS", "UP"]


def plot_training_curves(history, save_path):
    """Grafik loss & accuracy per epoch -> bukti visual wajib untuk laporan."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history.history["loss"], label="Training Loss")
    axes[0].plot(history.history["val_loss"], label="Validation Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Training Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[1].axhline(y=1/3, color="gray", linestyle="--", alpha=0.6, label="Random baseline (33.3%)")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    -> Grafik training tersimpan di: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Confusion matrix -> bukti visual wajib untuk laporan."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label Sebenarnya")
    ax.set_title("Confusion Matrix (Validation Set)")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    -> Confusion matrix tersimpan di: {save_path}")

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
    print("\n" + report)
    return report


def main():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"[1/5] Mengambil data historis {config.SYMBOL} timeframe {config.PRIMARY_TIMEFRAME} ...")
    raw_df = data_fetcher.fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, total_candles=config.TRAIN_CANDLES)
    raw_df.to_csv(f"{config.DATA_DIR}/history_{config.PRIMARY_TIMEFRAME}.csv")
    print(f"    -> {len(raw_df)} candle diambil")

    print("[2/5] Feature engineering + labeling tren ...")
    X, y, scaler, feat_df = prepare_training_data(raw_df=raw_df)
    print(f"    -> X shape: {X.shape}, y shape: {y.shape}")
    print(f"    -> Distribusi label (0=DOWN,1=SIDEWAYS,2=UP): {np.bincount(y)}")

    # split time-series aware (tidak diacak, karena data time series)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # atasi class imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))
    print(f"    -> Class weights: {class_weight}")

    print("[3/5] Membangun model neural network ...")
    model = build_model(input_shape=(X.shape[1], X.shape[2]))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
                      restore_best_weights=True),
        ModelCheckpoint(config.MODEL_PATH, monitor="val_accuracy", save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    print("[4/5] Training model (learning rate = 0.001) ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    print("[5/5] Evaluasi akhir + membuat bukti visual untuk laporan ...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"    -> Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

    # --- Simpan history training (untuk dokumentasi/laporan) ---
    with open(f"{config.LOG_DIR}/training_history.json", "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # --- Grafik loss & accuracy ---
    plot_training_curves(history, f"{config.LOG_DIR}/training_curves.png")

    # --- Confusion matrix di validation set ---
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    report = plot_confusion_matrix(y_val, y_pred, f"{config.LOG_DIR}/confusion_matrix.png")

    with open(f"{config.LOG_DIR}/classification_report.txt", "w") as f:
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n\n")
        f.write(report)

    print(f"\nModel tersimpan di: {config.MODEL_PATH}")
    print(f"Scaler tersimpan di: {config.SCALER_PATH}")
    print(f"Semua bukti visual (untuk laporan) tersimpan di folder: {config.LOG_DIR}/")
    print("  - training_curves.png  (grafik loss & accuracy)")
    print("  - confusion_matrix.png (confusion matrix)")
    print("  - classification_report.txt (precision/recall/f1 per kelas)")


if __name__ == "__main__":
    main()