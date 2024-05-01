from datasets import Dataset

from inference_pipeline.data import utils
from inference_pipeline.paths import DATA_PATH
from pathlib import Path


class DisasterTweetData:
    def __init__(
        self,
        data_path: Path,
    ):
        """Cette classe permet de récupérer et préparer les données d'entrainement ou d'inférence.
        Args:
            data_path(Path): le repertoire des données.

        """
        self._data_path = data_path

    def clean(self, text: str) -> dict:
        """Cette méthode applique différentes transformations à un texte.
        Returns:
            dict : un dictionnaire contenant le texte transformé.
        """
        return {"text": utils.clean_tweet(text)}

    def load_data(self) -> Dataset:
        """
        Cette méthode permet de préparer les données d'inférence.
        Elle crée un Dataset à partir d'un fichier csv.
        Returns:
            Dataset: un objet Dataset de HF.
        """

        path = (self._data_path / "test.csv").__str__()

        data = (
            Dataset.from_csv(path, split="test")
            .remove_columns(["id", "keyword", "location"])
            .map(self.clean, input_columns="text")
        )

        return data.shuffle()


if __name__ == "__main__":
    data_path = DATA_PATH
    print(DisasterTweetData(data_path=data_path).load_data().shuffle()[0])
