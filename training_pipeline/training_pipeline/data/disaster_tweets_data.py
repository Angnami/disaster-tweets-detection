from datasets import Dataset, DatasetDict

from training_pipeline.data import utils
from training_pipeline import constants
from training_pipeline.paths import DATA_PATH
from pathlib import Path
from typing import Optional


class DisasterTweetData:
    def __init__(
        self,
        data_path: Path,
        scope: constants.Scope = constants.Scope.TRAINING,
        test_size: Optional[int] = constants.VALIDATION_SIZE,
        seed: Optional[int] = constants.RANDOM_STATE,
    ):
        """Cette classe permet de récupérer et préparer les données d'entrainement ou d'inférence.
        Args:
            scope: la portée des données (entrainement ou inférence)
            test_size(optionnel): la taille des données de validation
            seed(optionnel): la graine pour la répartition aléatoire des données d'entrainement
        """
        self._scope = scope
        self._test_size = test_size
        self._seed = seed
        self._data_path = data_path

    def clean(self, text: str) -> dict:
        """Cette méthode applique différentes transformations à un texte.
        Returns:
            dict : un dictionnaire contenant le texte transformé.
        """
        return {"text": utils.clean_tweet(text)}

    def load_data(self) -> DatasetDict | Dataset:
        """
        Cette méthode permet de préparer les données d'entrainement ou d'inférence.
        Elle crée un Dataset ou DatasetDict à partir d'un fichier csv,retire les colonnes
        non nécessaires, applique un ensemble de transformations à la colone text,
        convertit la colonne target en string et ajoute ses modalités. Elle répartit aléatoirement
        les données d'entrainement entre l'entrainement et la validation.
        Returns:
            DatasetDict|Dataset: un objet DatasetDict ou Dataset de HF.
        """
        if self._scope == constants.Scope.TRAINING:
            path = (self._data_path / "train.csv").__str__()

            data = (
                Dataset.from_csv(path)
                .remove_columns(["id", "keyword", "location"])
                .rename_column("target", "label")
                .map(self.clean, input_columns="text")
                .train_test_split(test_size=self._test_size, seed=self._seed)
            )

            return data
        else:
            path = (self._data_path / "inference.csv").__str__()

            data = (
                Dataset.from_csv(path)
                .remove_columns(["id", "keyword", "location"])
                .rename_column("target", "label")
                .map(self.clean, input_columns="text")
            )

            return data


if __name__ == "__main__":
    data_path = DATA_PATH
    print(
            len(
                DisasterTweetData(
                data_path=data_path, scope=constants.Scope.TRAINING
            ).load_data()["train"]
              )
        )
