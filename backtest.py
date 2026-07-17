"""
backtest.py
Menguji performa sinyal model pada data historis (out-of-sample)
sebelum digunakan secara live. Menghitung win rate & simulasi PnL sederhana.

Cara pakai:
    python backtest.py
"""

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

import config
import data_fetcher
from indicators import add_technical_indicators
from dataset import FEATURE_COLUMNS, label_trend


def run_threshold_sweep(df, model, scaler, horizon=config.PREDICTION_HORIZON, fee_pct=0.05):
    """
    Coba beberapa CONFIDENCE_THRESHOLD sekaligus untuk melihat mana yang
    paling profitable, daripada menebak satu nilai secara manual.
    """
    features = df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)
    window = config.LOOKBACK_WINDOW

    # Kumpulkan semua prediksi sekali saja, filter threshold dilakukan belakangan
    all_predictions = []
    for i in range(window, len(features_scaled) - horizon):
        X = np.expand_dims(features_scaled[i - window:i], axis=0)
        probs = model.predict(X, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        entry_price = df["close"].iloc[i]
        exit_price = df["close"].iloc[i + horizon]
        pct_change = (exit_price - entry_price) / entry_price * 100

        all_predictions.append({
            "pred_class": pred_class,
            "confidence": confidence,
            "pct_change": pct_change,
        })

    print("\n===== THRESHOLD SWEEP (mencari threshold paling profitable) =====")
    print(f"{'Threshold':<10} {'Trades':<8} {'WinRate':<10} {'AvgPnL%':<10} {'TotalReturn%':<12}")

    best = {"threshold": None, "total_return": -999}

    for threshold in [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55]:
        equity = 1000.0
        wins = 0
        n_trades = 0
        pnl_list = []

        for p in all_predictions:
            if p["confidence"] < threshold or p["pred_class"] == 1:
                continue
            if p["pred_class"] == 2:
                pnl_pct = p["pct_change"] - fee_pct
            else:
                pnl_pct = -p["pct_change"] - fee_pct

            equity *= (1 + pnl_pct / 100)
            pnl_list.append(pnl_pct)
            n_trades += 1
            if pnl_pct > 0:
                wins += 1

        if n_trades > 0:
            win_rate = wins / n_trades * 100
            avg_pnl = np.mean(pnl_list)
            total_return = (equity / 1000.0 - 1) * 100
            print(f"{threshold:<10} {n_trades:<8} {win_rate:<10.1f} {avg_pnl:<10.3f} {total_return:<12.2f}")

            if total_return > best["total_return"]:
                best = {"threshold": threshold, "total_return": total_return,
                        "n_trades": n_trades, "win_rate": win_rate}
        else:
            print(f"{threshold:<10} {'0':<8} {'-':<10} {'-':<10} {'-':<12}")

    print("=" * 60)
    if best["threshold"] is not None:
        print(f"\n>> Threshold terbaik: {best['threshold']} "
              f"(return {best['total_return']:.2f}%, {best['n_trades']} trades, "
              f"win rate {best['win_rate']:.1f}%)")
        print(f">> Update CONFIDENCE_THRESHOLD di config.py ke {best['threshold']} untuk hasil optimal.")
    print("=" * 60 + "\n")

    return best


def run_backtest(total_candles: int = 2000, fee_pct: float = 0.05):
    print("Mengambil data historis untuk backtest ...")
    raw_df = data_fetcher.fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, total_candles)
    df = add_technical_indicators(raw_df)
    df["label"] = label_trend(df)
    df.dropna(inplace=True)

    model = tf.keras.models.load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    print(f"CONFIDENCE_THRESHOLD aktif di config.py: {config.CONFIDENCE_THRESHOLD}")

    # Jalankan sweep dulu untuk mencari threshold terbaik
    run_threshold_sweep(df, model, scaler)

    features = df[FEATURE_COLUMNS].values
    features_scaled = scaler.transform(features)

    window = config.LOOKBACK_WINDOW
    horizon = config.PREDICTION_HORIZON

    trades = []
    equity = 1000.0  # modal awal simulasi (USDT)
    equity_curve = [equity]
    skipped_by_regime = 0

    for i in range(window, len(features_scaled) - horizon):
        X = np.expand_dims(features_scaled[i - window:i], axis=0)
        probs = model.predict(X, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        if confidence < config.CONFIDENCE_THRESHOLD or pred_class == 1:
            continue  # skip sideways / low confidence

        # Filter regime: skip kalau market tidak sedang trending kuat (ADX rendah)
        if config.USE_REGIME_FILTER:
            current_adx = df["adx_14"].iloc[i]
            if pd.isna(current_adx) or current_adx < config.ADX_TREND_THRESHOLD:
                skipped_by_regime += 1
                continue

        entry_price = df["close"].iloc[i]
        exit_price = df["close"].iloc[i + horizon]
        pct_change = (exit_price - entry_price) / entry_price * 100

        if pred_class == 2:  # LONG
            pnl_pct = pct_change - fee_pct
        else:  # SHORT
            pnl_pct = -pct_change - fee_pct

        equity *= (1 + pnl_pct / 100)
        equity_curve.append(equity)

        trades.append({
            "timestamp": df.index[i],
            "direction": "LONG" if pred_class == 2 else "SHORT",
            "confidence": confidence,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
        })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        print("Tidak ada trade yang memenuhi threshold confidence.")
        return

    win_rate = (trades_df["pnl_pct"] > 0).mean()
    avg_pnl = trades_df["pnl_pct"].mean()
    total_return = (equity / 1000.0 - 1) * 100

    print("\n===== HASIL BACKTEST =====")
    print(f"Total trade         : {len(trades_df)}")
    print(f"Win rate             : {win_rate:.2%}")
    print(f"Rata-rata PnL/trade  : {avg_pnl:.2f}%")
    print(f"Total return          : {total_return:.2f}%")
    print(f"Equity akhir (mulai 1000) : {equity:.2f}")
    if config.USE_REGIME_FILTER:
        print(f"Sinyal di-skip karena regime tidak trending (ADX<{config.ADX_TREND_THRESHOLD}): {skipped_by_regime}")
    print("===========================")
    print("\nCatatan: backtest ini sederhana (tanpa slippage realistis, tanpa")
    print("liquidity constraint). Hasil historis TIDAK menjamin performa masa depan.")

    trades_df.to_csv(f"{config.LOG_DIR}/backtest_trades.csv", index=False)
    return trades_df


if __name__ == "__main__":
    run_backtest()