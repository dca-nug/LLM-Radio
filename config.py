# ============================================================
# config.py — sumber tunggal lokasi data (input + output).
# Semua script membaca/menulis di DATA_DIR.
# Atur lewat environment variable TB_DATA_DIR; default: ./data
#   Linux/macOS : export TB_DATA_DIR=/path/ke/data
#   Windows PS  : $env:TB_DATA_DIR = "F:\data\tb-cxr"
#   Windows CMD : set TB_DATA_DIR=F:\data\tb-cxr
# Data mentah TIDAK disertakan di repo (lihat data/README.md).
# ============================================================
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("TB_DATA_DIR", "data")).expanduser().resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
