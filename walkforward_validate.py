"""
walkforward_validate.py
Validasi out-of-sample yang lebih jujur secara statistik dibanding backtest
biasa: model di-training ulang di beberapa 'fold' dengan expanding window,
lalu diuji HANYA di data yang belum pernah dilihat model pada fold tsb.

Kenapa ini penting? backtest.py sebelumnya mencoba banyak threshold dan
memilih yang terbaik dari data yang SAMA -> rentan "data snooping" (hasil
bagus karena kebetulan, bukan model benar-benar punya edge). Di sini,
threshold diuji di data baru yang independen di tiap fold, dan kita lihat
apakah hasilnya KONSISTEN di banyak fold, bukan cuma bagus di satu tempat.

PERINGATAN: script ini training model beberapa kali, bisa memakan waktu
15-30 menit tergantung CPU Anda. Kalau ingin lebih cepat, kecilkan
N_FOLDS atau FOLD_EPOCHS di bagian atas file ini.

Cara pakai:
    python walkforward_validate.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

import config
import data_fetcher
from indicators import add_technical_indicators
from dataset import FEATURE_COLUMNS, label_trend
from model import build_model

# ============ PARAMETER WALK-FORWARD (bisa disesuaikan) ============
N_FOLDS = 4                             # jumlah fold out-of-sample
CANDLES_PER_TEST_FOLD = 400        # ukuran tiap fold pengujian (candle)
FOLD_EPOCHS = 40                       # epoch training per fold (dikurangi biar tidak kelamaan)
CONFIDENCE_THRESHOLDS_TO_TEST = [0.35, 0.38, 0.40, 0.42, 0.45]
FEE_PCT = 0.05


def create_sequences_from_array(features_scaled, labels, window):
    X, y = [], []
    for i in range(window, len(features_scaled)):
        X.append(features_scaled[i - window:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


def main():
    print("Mengambil data historis untuk walk-forward validation ...", flush=True)
    raw_df = data_fetcher.fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, config.TRAIN_CANDLES)
    df = add_technical_indicators(raw_df)
    df["label"] = label_trend(df)
    df.dropna(inplace=True)
    df = df.reset_index()
    print(f"Total data setelah cleaning: {len(df)} baris", flush=True)

    n = len(df)
    window = config.LOOKBACK_WINDOW
    horizon = config.PREDICTION_HORIZON

    fold_start = n - N_FOLDS * CANDLES_PER_TEST_FOLD
    if fold_start < window * 3:
        raise ValueError(
            "Data tidak cukup untuk walk-forward dengan setting ini. "
            "Kecilkan N_FOLDS/CANDLES_PER_TEST_FOLD, atau naikkan TRAIN_CANDLES di config.py."
        )

    results_per_threshold = {t: [] for t in CONFIDENCE_THRESHOLDS_TO_TEST}

    for fold in range(N_FOLDS):
        test_start = fold_start + fold * CANDLES_PER_TEST_FOLD
        test_end = test_start + CANDLES_PER_TEST_FOLD
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start - window: test_end]  # sertakan window awal utk sequence

        print(f"\n===== FOLD {fold + 1}/{N_FOLDS} =====", flush=True)
        print(f"Train: candle 0-{test_start} ({len(train_df)} candle) | "
              f"Test out-of-sample: candle {test_start}-{test_end}", flush=True)

        scaler = StandardScaler()
        train_features_scaled = scaler.fit_transform(train_df[FEATURE_COLUMNS].values)
        train_labels = train_df["label"].values
        X_train, y_train = create_sequences_from_array(train_features_scaled, train_labels, window)

        model = build_model(input_shape=(window, len(FEATURE_COLUMNS)))
        model.fit(
            X_train, y_train,
            epochs=FOLD_EPOCHS,
            batch_size=config.BATCH_SIZE,
            verbose=0,
            validation_split=0.1,
        )
        print("  Training fold selesai. Menguji di data out-of-sample ...", flush=True)

        test_features_scaled = scaler.transform(test_df[FEATURE_COLUMNS].values)
        close_prices = test_df["close"].values
        adx_values = test_df["adx_14"].values if "adx_14" in test_df.columns else None

        fold_predictions = []
        for i in range(window, len(test_features_scaled) - horizon):
            X = np.expand_dims(test_features_scaled[i - window:i], axis=0)
            probs = model.predict(X, verbose=0)[0]
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            entry_price = close_prices[i]
            exit_price = close_prices[i + horizon]
            pct_change = (exit_price - entry_price) / entry_price * 100
            current_adx = adx_values[i] if adx_values is not None else None
            fold_predictions.append((pred_class, confidence, pct_change, current_adx))

        for t in CONFIDENCE_THRESHOLDS_TO_TEST:
            equity = 1000.0
            wins = 0
            n_trades = 0
            for pred_class, confidence, pct_change, current_adx in fold_predictions:
                if confidence < t or pred_class == 1:
                    continue
                if config.USE_REGIME_FILTER:
                    if current_adx is None or np.isnan(current_adx) or current_adx < config.ADX_TREND_THRESHOLD:
                        continue
                pnl_pct = (pct_change if pred_class == 2 else -pct_change) - FEE_PCT
                equity *= (1 + pnl_pct / 100)
                n_trades += 1
                if pnl_pct > 0:
                    wins += 1
            ret = (equity / 1000.0 - 1) * 100
            wr = (wins / n_trades * 100) if n_trades else None
            results_per_threshold[t].append({"fold": fold + 1, "trades": n_trades, "win_rate": wr, "return": ret})
            wr_str = f"{wr:.1f}%" if wr is not None else "-"
            print(f"    Threshold {t}: {n_trades} trades, win_rate={wr_str}, return={ret:.2f}%", flush=True)

    print(f"\n\n===== RINGKASAN WALK-FORWARD (OUT-OF-SAMPLE, {N_FOLDS} fold) =====")
    print(f"{'Threshold':<10}{'TotalTrades':<13}{'AvgWinRate':<13}{'AvgReturn%':<13}{'FoldPositif':<12}")
    for t in CONFIDENCE_THRESHOLDS_TO_TEST:
        rows = results_per_threshold[t]
        total_trades = sum(r["trades"] for r in rows)
        valid_wr = [r["win_rate"] for r in rows if r["win_rate"] is not None]
        avg_wr = np.mean(valid_wr) if valid_wr else None
        avg_ret = np.mean([r["return"] for r in rows])
        positive_folds = sum(1 for r in rows if r["return"] > 0)
        avg_wr_str = f"{avg_wr:.1f}%" if avg_wr is not None else "-"
        print(f"{t:<10}{total_trades:<13}{avg_wr_str:<13}{avg_ret:<13.2f}{f'{positive_folds}/{N_FOLDS}':<12}")

    print("\nCara membaca hasil ini:")
    print("- 'FoldPositif' = berapa dari total fold yang returnnya positif.")
    print("- Threshold yang BENAR-BENAR bagus idealnya positif di MAYORITAS fold")
    print("  (misal 3/4 atau 4/4), bukan cuma rata-rata tinggi yang ditarik oleh 1 fold")
    print("  yang kebetulan bagus (itu tanda overfitting terhadap satu periode waktu).")
    print("- Kalau semua threshold sering negatif/tidak konsisten, itu artinya secara")
    print("  jujur model belum punya edge yang bisa diandalkan -> perlu perbaikan fitur/")
    print("  arsitektur yang lebih besar, bukan sekadar tuning threshold.")


if __name__ == "__main__":
    main()