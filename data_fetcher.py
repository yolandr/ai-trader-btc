"""
data_fetcher.py
Mengambil data candlestick (OHLCV) historis maupun real-time dari exchange
menggunakan library ccxt (mendukung Binance, Bybit, OKX, dll).
"""

import time
import warnings
import pandas as pd
import ccxt
import urllib3

import config

# WORKAROUND: sebagian jaringan/laptop (biasanya yang dikelola sekolah/kantor)
# melakukan SSL interception sehingga verifikasi sertifikat gagal walau koneksi
# sebenarnya aman. Set DISABLE_SSL_VERIFY = True di config.py HANYA jika Anda
# sudah mengonfirmasi penyebabnya adalah SSL interception di jaringan/device
# Anda sendiri (bukan indikasi serangan MITM oleh pihak tidak dikenal).
DISABLE_SSL_VERIFY = getattr(config, "DISABLE_SSL_VERIFY", False)
if DISABLE_SSL_VERIFY:
    warnings.warn(
        "SSL verification DINONAKTIFKAN untuk koneksi exchange. "
        "Gunakan hanya jika Anda memahami risikonya dan hanya untuk data publik.",
        stacklevel=2,
    )
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_exchange():
    """Membuat instance exchange ccxt."""
    exchange_class = getattr(ccxt, config.EXCHANGE_ID)
    exchange = exchange_class({
        "enableRateLimit": True,
        "timeout": 30000,  # 30 detik, lebih toleran untuk koneksi lambat
    })
    if DISABLE_SSL_VERIFY:
        # ccxt punya property 'verify' sendiri yang dipakai saat request,
        # terpisah dari requests.Session.verify -> keduanya perlu di-set.
        exchange.verify = False
        exchange.session.verify = False
    return exchange


def _retry_call(func, retries: int = 3, delay: int = 5):
    """Bungkus pemanggilan API dengan retry otomatis kalau koneksi gagal/timeout."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except Exception as e:
            last_err = e
            print(f"[WARN] Percobaan {attempt}/{retries} gagal: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise last_err


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000, since: int = None) -> pd.DataFrame:
    """
    Ambil data OHLCV dan kembalikan sebagai DataFrame dengan kolom:
    timestamp, open, high, low, close, volume
    """
    exchange = get_exchange()
    raw = _retry_call(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since))
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df


def fetch_full_history(symbol: str, timeframe: str, total_candles: int = 5000) -> pd.DataFrame:
    """
    Ambil data historis lebih panjang dengan looping (karena exchange biasanya
    membatasi 1000 candle per request).
    """
    exchange = get_exchange()
    all_data = []
    limit = 1000
    since = None
    fetched = 0

    while fetched < total_candles:
        batch = _retry_call(lambda: exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since))
        if not batch:
            break
        all_data = batch + all_data
        since = batch[0][0] - (limit * _timeframe_to_ms(timeframe))
        fetched += len(batch)
        time.sleep(exchange.rateLimit / 1000)
        if len(batch) < limit:
            break

    df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df.tail(total_candles)


def _timeframe_to_ms(timeframe: str) -> int:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
    return value * mult[unit]


def fetch_latest_candle(symbol: str, timeframe: str) -> pd.Series:
    """Ambil candle terbaru (belum tentu closed) untuk real-time monitoring."""
    df = fetch_ohlcv(symbol, timeframe, limit=2)
    return df.iloc[-1]


if __name__ == "__main__":
    df = fetch_full_history(config.SYMBOL, config.PRIMARY_TIMEFRAME, total_candles=2000)
    print(df.tail())
    df.to_csv(f"{config.DATA_DIR}/{config.SYMBOL.replace('/', '')}_{config.PRIMARY_TIMEFRAME}.csv")
    print(f"Data tersimpan: {len(df)} candle")