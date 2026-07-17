# Panduan Setup Project di Laptop Baru

Panduan ini merangkum semua langkah supaya project langsung jalan tanpa perlu
mengulang troubleshooting yang sudah pernah ditemukan sebelumnya.

## 0. Persiapan Sebelum Mulai

**WAJIB dicek dulu sebelum instalasi apapun:**

- [ ] Punya **VPN** yang aktif & berfungsi (Windscribe/ProtonVPN gratis cukup).
      Ini wajib karena **Binance memblokir akses dari IP Indonesia** (error 451),
      dan sebagian ISP Indonesia (mis. Indosat) juga memblokir Binance sendiri.
      Tanpa VPN aktif, program TIDAK akan bisa mengambil data sama sekali.
- [ ] Python **BUKAN dari Microsoft Store** — download dari python.org.
      Python Store menyebabkan error "long path" saat install TensorFlow.

## 1. Install Python (skip kalau sudah punya python.org version)

1. Buka **https://www.python.org/downloads/**
2. Download Python 3.11 (bukan versi terbaru banget, TensorFlow kadang belum
   support versi Python paling baru — 3.11 paling aman)
3. Saat instalasi, **WAJIB centang "Add python.exe to PATH"**
4. Cek berhasil dengan buka PowerShell baru:
   ```powershell
   python --version
   where python
   ```
   Pastikan hasil `where python` **TIDAK** mengarah ke folder
   `WindowsApps` atau `PythonSoftwareFoundation...` (itu tanda Python Store).

## 2. Aktifkan Long Path Support di Windows (jaga-jaga)

Buka **PowerShell as Administrator** (klik kanan PowerShell di Start Menu ->
Run as Administrator), lalu jalankan:
```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
Lalu **restart komputer**.

## 3. Taruh Project di Folder dengan Path Pendek

**Jangan** taruh project di folder dengan nama panjang/ada spasi (misal
`D:\Kuliah\Semester 4\Tugas AI\...`). Pakai path pendek, contoh:
```
C:\btc-ai-trader\
```
Extract semua file project (`config.py`, `train.py`, dll) ke folder ini.

## 4. Buat Virtual Environment & Install Dependencies

Buka PowerShell (tidak perlu admin), masuk ke folder project:
```powershell
cd C:\btc-ai-trader
python -m venv venv
venv\Scripts\activate
```
Setelah aktif, akan muncul `(venv)` di depan prompt. Lalu install semua
library yang dibutuhkan:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Proses ini bisa memakan waktu beberapa menit (TensorFlow ukurannya besar).

## 5. Nyalakan VPN, lalu Tes Koneksi ke Binance

**Nyalakan VPN dulu** (pilih server luar Indonesia, misal Singapore), baru tes:
```powershell
python -c "import requests; r = requests.get('https://api.binance.com/api/v3/ping', timeout=15); print(r.status_code, r.text)"
```
Harus keluar `200 {}`. Kalau muncul error/451/SSL error, VPN belum aktif dengan
benar — jangan lanjut ke langkah berikutnya sebelum ini beres.

## 6. Jalankan Project

Urutan yang benar (VPN harus tetap aktif di setiap langkah):

```powershell
# 1. Training model dari nol (wajib pertama kali)
python train.py

# 2. Cek hasil training di folder logs/ -> training_curves.png, confusion_matrix.png

# 3. Backtest untuk lihat performa trading
python backtest.py

# 4. (Opsional, lebih lama) Validasi out-of-sample yang lebih jujur
python walkforward_validate.py

# 5. Jalankan monitoring real-time
python realtime_predictor.py
```

## Troubleshooting Cepat (kalau masih error)

| Gejala | Kemungkinan Penyebab | Solusi |
|---|---|---|
| `OSError ... No such file or directory` saat `pip install` | Path kepanjangan / Python Store | Pindah ke `C:\` + install Python dari python.org |
| `SSLCertVerificationError` | Interceptor SSL di laptop/jaringan (jarang) | Cek `netsh winhttp show proxy`, atau hubungi saya lagi |
| Error `451 Service unavailable from a restricted location` | VPN mati/belum aktif | Nyalakan VPN, tes ulang langkah 5 |
| `ConnectTimeout` ke Binance | VPN mati / jaringan lambat | Cek VPN, coba server lain |
| `python` tidak dikenali di PowerShell | Python belum ditambah ke PATH | Install ulang Python, centang "Add to PATH" |

## Catatan Penting

- **VPN harus selalu aktif** setiap kali menjalankan script apapun yang
  mengambil data (`train.py`, `backtest.py`, `walkforward_validate.py`,
  `realtime_predictor.py`, `diagnose_model.py`).
- Jangan install `pip-system-certs` — package ini pernah menyebabkan
  konflik SSL yang sulit didiagnosis di project ini.
- Folder `venv/`, `__pycache__/`, `data/`, `models/`, `logs/` akan otomatis
  terbentuk saat dipakai — tidak perlu dibuat manual.
