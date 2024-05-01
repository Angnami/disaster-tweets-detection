from pathlib import Path

path = Path(
    "/home/angnami/.cache/disaster-tweets-detection/models/angnami/disaster-tweets-detection-distilbert:1.0.0"
).resolve()

sub_dirs = [d for d in path.iterdir() if d.is_dir()]
print(len(sub_dirs))
