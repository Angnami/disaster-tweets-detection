from pathlib import Path

# --- CACHING ----
CACHE_DIR = CACHE_DIR = Path.home() / ".cache" / "disaster-tweets-detection"

# ---- MODEL -----

FINE_TUNED_MODEL_CHKPT = "angnami/disaster-tweets-detection-distilbert:1.0.0"
