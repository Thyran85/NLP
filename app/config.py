from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

CACHE_DIR = Path(os.getenv("HF_HOME", str(BASE_DIR / ".model-cache")))
if not CACHE_DIR.is_absolute():
    CACHE_DIR = BASE_DIR / CACHE_DIR
os.environ["HF_HOME"] = str(CACHE_DIR)
