from pathlib import Path

PARENT_DIR = Path(__file__).parent.resolve().parent.resolve()


DATA_PATH = PARENT_DIR / "dataset"

Path.mkdir(DATA_PATH, parents=True, exist_ok=True)

ZIP_FILE_PATH = PARENT_DIR / "nlp-getting-started.zip"
