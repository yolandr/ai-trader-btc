"""
dataset.py
Menyiapkan data training: feature engineering, labeling tren,
normalisasi, dan pembuatan sequence untuk LSTM.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

import config
from indicators import add_technical_indicators

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "ema_9", "ema_21", "ema_50",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width", "bb_percent_b",
    "atr_14", "volume_ratio", "returns", "volatility_10",
    "adx_14", "roc_10", "stoch_k", "stoch_d", "obv_norm", "volatility_regime",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def label_trend(df: pd.DataFrame, horizon: int = config.PREDICTION_HORIZON,
                 atr_multiplier: float = config.ATR_LABEL_MULTIPLIER) -> pd.Series:
    """
    Label 0 = DOWN, 1 = SIDEWAYS, 2 = UP
    berdasarkan perubahan harga 'horizon' candle ke depan, dibandingkan
    dengan threshold DINAMIS = atr_multiplier * (ATR/close) * 100.

    Kenapa dinamis (bukan persentase tetap)? Volatilitas BTC berubah-ubah;
    saat volatilitas tinggi, pergerakan 0.5% bisa jadi "noise" biasa, sementara
    saat volatilitas rendah, 0.5% bisa jadi pergerakan signifikan. Threshold
    berbasis ATR membuat label lebih konsisten secara statistik di berbagai
    kondisi market.
    """
    future_price = df["close"].shift(-horizon)
    pct_change = (future_price - df["close"]) / df["close"] * 100

    # threshold dinamis per baris, dengan batas minimum agar tidak terlalu sensitif
    dynamic_threshold = (df["atr_14"] / df["close"]) * 100 * atr_multiplier
    dynamic_threshold = dynamic_threshold.clip(lower=config.MIN_LABEL_THRESHOLD_PCT)

    labels = pd.Series(1, index=df.index)  # default SIDEWAYS
    labels[pct_change > dynamic_threshold] = 2   # UP
    labels[pct_change < -dynamic_threshold] = 0  # DOWN
    return labels


def build_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = add_technical_indicators(raw_df)
    df["label"] = label_trend(df)
    df.dropna(inplace=True)
    return df


def create_sequences(df: pd.DataFrame, window: int = config.LOOKBACK_WINDOW, fit_scaler: bool = True,
                      scaler: StandardScaler = None):
    """
    Mengubah dataframe menjadi sequence 3D (samples, timesteps, features)
    untuk input LSTM, plus label per sequence.
    """
    features = df[FEATURE_COLUMNS].values
    labels = df["label"].values

    if fit_scaler:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        joblib.dump(scaler, config.SCALER_PATH)
    else:
        features_scaled = scaler.transform(features)

    X, y = [], []
    for i in range(window, len(features_scaled)):
        X.append(features_scaled[i - window:i])
        y.append(labels[i])

    return np.array(X), np.array(y), scaler


def prepare_training_data(csv_path: str = None, raw_df: pd.DataFrame = None):
    """Pipeline lengkap: raw data -> features -> sequences siap training."""
    if raw_df is None:
        raw_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    feat_df = build_feature_dataframe(raw_df)
    X, y, scaler = create_sequences(feat_df, fit_scaler=True)

    return X, y, scaler, feat_df


if __name__ == "__main__":
    import data_fetcher
    raw = data_fetcher.fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, 3000)
    X, y, scaler, feat_df = prepare_training_data(raw_df=raw)
    print("Shape X:", X.shape)   # (samples, window, n_features)
    print("Shape y:", y.shape)
    print("Distribusi label:", np.bincount(y))
