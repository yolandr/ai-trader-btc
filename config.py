"""
Konfigurasi utama untuk BTC AI Trading Bot
"""

# ============ EXCHANGE & DATA ============
EXCHANGE_ID = "binance"          # exchange via ccxt
SYMBOL = "BTC/USDT"
TIMEFRAMES = ["15m", "1h", "4h"] # multi-timeframe analysis
PRIMARY_TIMEFRAME = "1h"          # timeframe utama untuk sinyal
HISTORY_LIMIT = 1000               # jumlah candle historis yang diambil per fetch

# WORKAROUND untuk jaringan/laptop dengan SSL interception (mis. laptop sekolah/
# kantor dengan security agent). Set True HANYA setelah Anda mengonfirmasi
# penyebab error SSL adalah interception di device/jaringan Anda sendiri.
DISABLE_SSL_VERIFY = False

# ============ FEATURE ENGINEERING ============
LOOKBACK_WINDOW = 60        # jumlah candle sebelumnya yang dilihat model (sequence length)
PREDICTION_HORIZON = 5       # prediksi tren N candle ke depan
SR_LOOKBACK = 100              # jumlah candle untuk deteksi support/resistance
SR_TOLERANCE_PCT = 0.15        # toleransi (%) untuk mengelompokkan level S/R yang berdekatan
SWING_ORDER = 5                 # sensitivitas deteksi swing high/low (semakin besar = semakin halus)

# Label tren dinamis berbasis ATR (volatilitas), bukan persentase tetap.
# threshold = ATR_LABEL_MULTIPLIER * (ATR/close) * 100, dengan batas minimum.
ATR_LABEL_MULTIPLIER = 0.8
MIN_LABEL_THRESHOLD_PCT = 0.25
TREND_THRESHOLD_PCT = 0.5   # tetap disimpan untuk kompatibilitas/referensi lama

# ============ TRAINING DATA ============
TRAIN_CANDLES = 8000   # dinaikkan dari 5000 -> lebih banyak pola historis

# ============ NEURAL NETWORK ============
LEARNING_RATE = 0.001     # sesuai permintaan
BATCH_SIZE = 64
EPOCHS = 150
LSTM_UNITS = [96, 48, 24]     # sedikit disederhanakan untuk mengurangi overfitting
DENSE_UNITS = [24, 12]
DROPOUT_RATE = 0.4               # dinaikkan dari 0.3 -> regularisasi lebih kuat
NUM_CLASSES = 3                  # 0 = DOWN, 1 = SIDEWAYS, 2 = UP
EARLY_STOPPING_PATIENCE = 8   # diturunkan dari 20 -> berhenti lebih cepat saat val_loss stagnan
L2_REG = 1e-4                       # dinaikkan dari 1e-5 di model.py

# ============ PATHS ============
MODEL_DIR = "models"
DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_PATH = f"{MODEL_DIR}/btc_trend_model.keras"
SCALER_PATH = f"{MODEL_DIR}/scaler.pkl"

# ============ REAL-TIME ============
REALTIME_POLL_INTERVAL_SEC = 30   # seberapa sering cek market (detik)
CONFIDENCE_THRESHOLD = 0.40         # disesuaikan berdasarkan distribusi confidence model
                                       # (lihat diagnose_model.py -> max confidence model ini ~0.58)

# Filter regime: cuma ambil sinyal saat market sedang TRENDING kuat (ADX tinggi),
# skip saat market choppy/sideways (ADX rendah). ADX < 20 umumnya dianggap
# "tidak ada tren jelas" di analisis teknikal.
USE_REGIME_FILTER = True
ADX_TREND_THRESHOLD = 20