from training_pipeline import paths
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from training_pipeline.constants import RANDOM_STATE, TEST_SIZE


dataset_path = paths.DATA_PATH
zip_file_path = paths.ZIP_FILE_PATH


def extract_data() -> None:
    """
    Cette fonction extrait le contenu du fichier zip dans le repertoire des données dataset_path.
    Elle répartit aléatoirement les données entre l'entrainement et l'inférence.
    """
    with ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(members=["train.csv", "test.csv"], path=dataset_path)

    train_df = pd.read_csv(dataset_path / "train.csv")

    columns = train_df.columns.to_list()

    X_train, X_test, y_train, y_test = train_test_split(
        train_df.drop("target", axis=1).values,
        train_df.target.values,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    pd.DataFrame(
        np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1), columns=columns
    ).to_csv(dataset_path / "train.csv", index=False)
    pd.DataFrame(
        np.concatenate([X_test, y_test.reshape(-1, 1)], axis=1), columns=columns
    ).to_csv(dataset_path / "inference.csv", index=False)


if __name__ == "__main__":
    extract_data()
