"""
indicators.py
Menghitung indikator teknikal (RSI, MACD, EMA, ATR, Bollinger Bands)
dan mendeteksi level Support & Resistance dari swing high/low.
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

import config


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # EMA
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20

    # ATR (volatilitas, 14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()

    # Volume relatif
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-9)

    # Return & volatilitas
    df["returns"] = df["close"].pct_change()
    df["volatility_10"] = df["returns"].rolling(10).std()

    # --- Fitur tambahan: konteks tren & regime pasar ---

    # ADX (Average Directional Index) -> kekuatan tren, bukan arah
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / (atr_smooth + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / 14, adjust=False).mean() / (atr_smooth + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df["adx_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Rate of Change (momentum di beberapa periode)
    df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100

    # Stochastic Oscillator
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = (df["close"] - low_14) / (high_14 - low_14 + 1e-9) * 100
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # On-Balance Volume (dinormalisasi dengan rolling mean supaya skalanya stabil)
    obv_raw = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv_norm"] = (obv_raw - obv_raw.rolling(50).mean()) / (obv_raw.rolling(50).std() + 1e-9)

    # Posisi harga relatif di dalam Bollinger Bands (%B)
    df["bb_percent_b"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Volatilitas relatif (dibanding rata-rata 100 candle terakhir)
    # -> membantu model membedakan regime volatilitas tinggi vs rendah
    df["volatility_regime"] = df["atr_14"] / (df["atr_14"].rolling(100).mean() + 1e-9)

    # Fitur waktu siklikal (crypto trading 24/7, kadang ada pola per jam/hari)
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    return df


# ============================================================
# SUPPORT & RESISTANCE DETECTION
# ============================================================

def detect_swing_points(df: pd.DataFrame, order: int = config.SWING_ORDER):
    """
    Deteksi swing high & swing low menggunakan local extrema.
    order = jumlah candle di kiri-kanan yang harus lebih rendah/tinggi.
    """
    highs = df["high"].values
    lows = df["low"].values

    high_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    low_idx = argrelextrema(lows, np.less_equal, order=order)[0]

    swing_highs = df.iloc[high_idx][["high"]].rename(columns={"high": "price"})
    swing_lows = df.iloc[low_idx][["low"]].rename(columns={"low": "price"})

    return swing_highs, swing_lows


def cluster_levels(prices: pd.Series, tolerance_pct: float = config.SR_TOLERANCE_PCT):
    """
    Mengelompokkan level-level harga yang berdekatan (dalam tolerance %)
    menjadi satu zona support/resistance, lalu urutkan berdasarkan
    berapa kali level itu "disentuh" (strength).
    """
    if len(prices) == 0:
        return []

    values = sorted(prices.tolist())
    clusters = []
    current_cluster = [values[0]]

    for price in values[1:]:
        if abs(price - current_cluster[-1]) / current_cluster[-1] * 100 <= tolerance_pct:
            current_cluster.append(price)
        else:
            clusters.append(current_cluster)
            current_cluster = [price]
    clusters.append(current_cluster)

    levels = []
    for c in clusters:
        levels.append({
            "price": round(float(np.mean(c)), 2),
            "strength": len(c),   # berapa kali level ini disentuh -> semakin tinggi semakin kuat
        })

    levels.sort(key=lambda x: x["strength"], reverse=True)
    return levels


def get_support_resistance(df: pd.DataFrame, lookback: int = config.SR_LOOKBACK):
    """
    Mengembalikan level support & resistance terkini relatif terhadap
    harga saat ini, lengkap dengan strength (jumlah sentuhan).
    """
    recent = df.tail(lookback)
    current_price = df["close"].iloc[-1]

    swing_highs, swing_lows = detect_swing_points(recent)

    resistance_levels = cluster_levels(swing_highs["price"])
    support_levels = cluster_levels(swing_lows["price"])

    # Pisahkan berdasarkan posisi relatif ke harga sekarang
    resistance_above = [lv for lv in resistance_levels if lv["price"] > current_price]
    support_below = [lv for lv in support_levels if lv["price"] < current_price]

    resistance_above.sort(key=lambda x: x["price"])   # terdekat dulu
    support_below.sort(key=lambda x: x["price"], reverse=True)  # terdekat dulu

    nearest_resistance = resistance_above[0] if resistance_above else None
    nearest_support = support_below[0] if support_below else None

    return {
        "current_price": round(float(current_price), 2),
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "all_support_levels": support_below[:5],
        "all_resistance_levels": resistance_above[:5],
    }


if __name__ == "__main__":
    import data_fetcher
    df = data_fetcher.fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, 500)
    df = add_technical_indicators(df)
    sr = get_support_resistance(df)
    print(sr)
