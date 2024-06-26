from enum import Enum
from pathlib import Path


class Scope(Enum):
    TRAINING = "training"

    INFERENCE = "inference"


RANDOM_STATE = 2024

TEST_SIZE = 0.1

VALIDATION_SIZE = 0.2

CACHE_DIR = Path.home() / ".cache" / "disaster-tweets-detection"

NUM_LABELS = 2

ID2LABEL = {0: "non-disaster", 1: "disaster"}

LABEL2ID = {"disaster": 1, "non-disaster": 0}

FILES_TO_SAVE_AFTER_TRAINING = [
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "vocab.txt",
]
