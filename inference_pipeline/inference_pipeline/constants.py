from pathlib import Path

#--- CACHING ----
CACHE_DIR = Path.home() / ".cache" / "disaster-tweets"

#--- DATA ----
NUM_LABELS = 2

ID2LABEL = {0: "non-disaster", 1: "disaster"}

LABEL2ID = {"disaster": 1, "non-disaster": 0}

#---- MODEL -----
MODEL_ID = "id: distilbert/distilbert-base-uncased-finetuned-sst-2-english"
FINE_TUNED_MODEL_CHKPT = "angnami/distilbert-base-uncased-finetuned-sst-2-english:1.0.0 "