"""
realtime_predictor.py
Memantau market BTC secara real-time (polling), menjalankan model AI untuk
memprediksi tren, dan menghasilkan penjelasan detail entry (support/resistance,
arah sell/long, confidence).

Cara pakai:
    python realtime_predictor.py
"""

import time
import datetime
import numpy as np
import joblib
import tensorflow as tf

import config
import data_fetcher
from indicators import add_technical_indicators, get_support_resistance
from dataset import FEATURE_COLUMNS

LABEL_MAP = {0: "DOWN (Bearish)", 1: "SIDEWAYS (Netral)", 2: "UP (Bullish)"}


class RealtimeAnalyzer:
    def __init__(self):
        print("Memuat model & scaler ...")
        self.model = tf.keras.models.load_model(config.MODEL_PATH)
        self.scaler = joblib.load(config.SCALER_PATH)

    def get_prediction(self, df):
        """Ambil sequence terakhir, prediksi probabilitas tren."""
        features = df[FEATURE_COLUMNS].values[-config.LOOKBACK_WINDOW:]
        features_scaled = self.scaler.transform(features)
        X = np.expand_dims(features_scaled, axis=0)  # (1, window, n_features)

        probs = self.model.predict(X, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        return pred_class, confidence, probs

    def build_signal(self, df):
        """Gabungkan prediksi AI + analisa support/resistance jadi sinyal entry."""
        pred_class, confidence, probs = self.get_prediction(df)
        sr = get_support_resistance(df)

        current_price = sr["current_price"]
        support = sr["nearest_support"]
        resistance = sr["nearest_resistance"]

        trend_label = LABEL_MAP[pred_class]
        current_adx = df["adx_14"].iloc[-1] if "adx_14" in df.columns else None
        is_trending = (current_adx is not None) and (current_adx >= config.ADX_TREND_THRESHOLD)

        # Tentukan rekomendasi posisi
        recommendation = "WAIT / NO CLEAR SIGNAL"
        reasoning = []

        regime_ok = (not config.USE_REGIME_FILTER) or is_trending

        if confidence >= config.CONFIDENCE_THRESHOLD and not regime_ok:
            recommendation = "WAIT / MARKET CHOPPY"
            reasoning.append(
                f"Confidence model cukup ({confidence:.1%}), tapi ADX saat ini "
                f"({current_adx:.1f} < {config.ADX_TREND_THRESHOLD}) menunjukkan market "
                f"sedang sideways/tidak trending -> sinyal ditahan untuk hindari whipsaw."
            )
        elif confidence >= config.CONFIDENCE_THRESHOLD and regime_ok:
            if pred_class == 2:  # UP -> potensi LONG
                recommendation = "LONG (BUY)"
                reasoning.append(f"Model memprediksi tren naik dengan confidence {confidence:.1%}.")
                if config.USE_REGIME_FILTER:
                    reasoning.append(f"Market sedang trending (ADX={current_adx:.1f}), sinyal lebih dipercaya.")
                if support:
                    reasoning.append(
                        f"Entry ideal di dekat support {support['price']} "
                        f"(strength: {support['strength']}x sentuhan), stop-loss di bawah support tsb."
                    )
                if resistance:
                    reasoning.append(f"Target take-profit di resistance terdekat: {resistance['price']}.")

            elif pred_class == 0:  # DOWN -> potensi SHORT/SELL
                recommendation = "SHORT (SELL)"
                reasoning.append(f"Model memprediksi tren turun dengan confidence {confidence:.1%}.")
                if resistance:
                    reasoning.append(
                        f"Entry ideal di dekat resistance {resistance['price']} "
                        f"(strength: {resistance['strength']}x sentuhan), stop-loss di atas resistance tsb."
                    )
                if support:
                    reasoning.append(f"Target take-profit di support terdekat: {support['price']}.")
            else:
                reasoning.append("Market diprediksi sideways, sebaiknya tunggu breakout jelas.")
        else:
            reasoning.append(
                f"Confidence model ({confidence:.1%}) di bawah threshold "
                f"({config.CONFIDENCE_THRESHOLD:.0%}), sinyal tidak cukup kuat."
            )

        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "symbol": config.SYMBOL,
            "timeframe": config.PRIMARY_TIMEFRAME,
            "current_price": current_price,
            "predicted_trend": trend_label,
            "confidence": round(confidence, 4),
            "probabilities": {
                "DOWN": round(float(probs[0]), 4),
                "SIDEWAYS": round(float(probs[1]), 4),
                "UP": round(float(probs[2]), 4),
            },
            "nearest_support": support,
            "nearest_resistance": resistance,
            "recommendation": recommendation,
            "reasoning": reasoning,
        }

    def print_signal(self, signal: dict):
        print("\n" + "=" * 60)
        print(f" {signal['symbol']} | {signal['timeframe']} | {signal['timestamp']}")
        print("=" * 60)
        print(f" Harga saat ini      : {signal['current_price']}")
        print(f" Prediksi tren        : {signal['predicted_trend']} (confidence {signal['confidence']:.1%})")
        print(f" Probabilitas         : DOWN={signal['probabilities']['DOWN']:.1%}  "
              f"SIDEWAYS={signal['probabilities']['SIDEWAYS']:.1%}  UP={signal['probabilities']['UP']:.1%}")
        if signal["nearest_support"]:
            print(f" Support terdekat      : {signal['nearest_support']['price']} "
                  f"(strength {signal['nearest_support']['strength']}x)")
        if signal["nearest_resistance"]:
            print(f" Resistance terdekat   : {signal['nearest_resistance']['price']} "
                  f"(strength {signal['nearest_resistance']['strength']}x)")
        print(f" REKOMENDASI          : {signal['recommendation']}")
        print(" Alasan:")
        for r in signal["reasoning"]:
            print(f"   - {r}")
        print("=" * 60)


def run_loop():
    analyzer = RealtimeAnalyzer()
    print(f"Mulai monitoring real-time {config.SYMBOL} setiap {config.REALTIME_POLL_INTERVAL_SEC} detik ...")
    print("(Tekan Ctrl+C untuk berhenti)")

    while True:
        try:
            # Ambil lebih banyak candle: beberapa indikator butuh rolling window
            # panjang (mis. volatility_regime pakai rolling 100), jadi perlu buffer
            # ekstra supaya setelah dropna() masih tersisa >= LOOKBACK_WINDOW baris.
            raw_df = data_fetcher.fetch_ohlcv(
                config.SYMBOL, config.PRIMARY_TIMEFRAME,
                limit=config.LOOKBACK_WINDOW + 150
            )
            df = add_technical_indicators(raw_df)
            df.dropna(inplace=True)

            if len(df) < config.LOOKBACK_WINDOW:
                print(f"[WARN] Data belum cukup ({len(df)}/{config.LOOKBACK_WINDOW} baris) "
                      f"setelah dropna, skip iterasi ini.")
                time.sleep(config.REALTIME_POLL_INTERVAL_SEC)
                continue

            signal = analyzer.build_signal(df)
            analyzer.print_signal(signal)

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(config.REALTIME_POLL_INTERVAL_SEC)


if __name__ == "__main__":
    run_loop()