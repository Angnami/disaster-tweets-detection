from pathlib import Path


if __name__ == "__main__":
    print(Path("disaster-tweet/inference_pipeline/streamlit-app/data/tweets.csv").is_file())
    