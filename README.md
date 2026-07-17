# BTC AI Trend Predictor & Support/Resistance Bot

Bot analisa BTC menggunakan neural network (LSTM multi-layer) untuk memprediksi
arah tren (UP/DOWN/SIDEWAYS), dikombinasikan dengan deteksi level support &
resistance otomatis untuk membantu menentukan titik entry LONG/SHORT.

## ⚠️ Disclaimer
Ini adalah tools riset & edukasi, **bukan financial advice**. Trading crypto
berisiko tinggi. Selalu gunakan manajemen risiko (stop-loss, position sizing)
dan jangan pernah trading dengan dana yang tidak siap Anda rugikan.

## 1. Instalasi

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Struktur Project

| File | Fungsi |
|---|---|
| `config.py` | Semua parameter (learning rate, timeframe, threshold) |
| `data_fetcher.py` | Ambil data OHLCV historis & real-time via ccxt (Binance) |
| `indicators.py` | Hitung EMA/RSI/MACD/ATR/BB + deteksi support & resistance |
| `dataset.py` | Feature engineering, labeling tren, pembuatan sequence |
| `model.py` | Arsitektur neural network (3-layer LSTM + Dense) |
| `train.py` | Training model dari data historis |
| `realtime_predictor.py` | Monitoring real-time + generate sinyal entry |
| `backtest.py` | Uji performa strategi di data historis |

## 3. Cara Menjalankan (step by step)

### Step 1 — Training model
```bash
python train.py
```
Ini akan:
1. Mengambil ±5000 candle historis BTC/USDT timeframe 1h dari Binance
2. Menghitung indikator teknikal & label tren (UP/DOWN/SIDEWAYS)
3. Melatih model LSTM dengan learning rate `0.001` (Adam optimizer)
4. Menyimpan model ke `models/btc_trend_model.keras` dan scaler ke `models/scaler.pkl`

### Step 2 — Backtest (WAJIB sebelum live)
```bash
python backtest.py
```
Menguji sinyal model pada data historis, menghasilkan win rate & simulasi PnL.
**Jangan lanjut ke real-time sebelum win rate & return backtest masuk akal.**

### Step 3 — Real-time monitoring
```bash
python realtime_predictor.py
```
Bot akan polling market setiap 30 detik (bisa diubah di `config.py`), lalu
mencetak analisa seperti:

```
============================================================
 BTC/USDT | 1h | 2026-07-07T10:30:00
============================================================
 Harga saat ini      : 68450.5
 Prediksi tren        : UP (Bullish) (confidence 68.2%)
 Probabilitas         : DOWN=12.1%  SIDEWAYS=19.7%  UP=68.2%
 Support terdekat      : 67800.0 (strength 3x)
 Resistance terdekat   : 69200.0 (strength 5x)
 REKOMENDASI          : LONG (BUY)
 Alasan:
   - Model memprediksi tren naik dengan confidence 68.2%.
   - Entry ideal di dekat support 67800.0 (strength: 3x sentuhan), stop-loss di bawah support tsb.
   - Target take-profit di resistance terdekat: 69200.0.
============================================================
```

## 4. Kustomisasi Penting di `config.py`

- `TIMEFRAMES` / `PRIMARY_TIMEFRAME` — ganti timeframe analisa (15m/1h/4h/1d)
- `LOOKBACK_WINDOW` — berapa candle historis dilihat model per prediksi
- `PREDICTION_HORIZON` — prediksi tren berapa candle ke depan
- `TREND_THRESHOLD_PCT` — sensitivitas label UP/DOWN vs SIDEWAYS
- `LEARNING_RATE = 0.001` — learning rate model (sesuai requirement)
- `CONFIDENCE_THRESHOLD` — minimal confidence agar sinyal dianggap valid

## 5. Rencana Pengembangan Lanjutan (opsional)
- Multi-timeframe fusion (gabungkan prediksi 15m + 1h + 4h)
- Ganti LSTM dengan Transformer/Attention untuk akurasi lebih baik
- Tambahkan notifikasi Telegram/Discord saat sinyal valid muncul
- Deploy sebagai service dengan Docker + scheduler (cron/Celery)
- Paper-trading otomatis via exchange API (testnet dulu!) sebelum live trading
