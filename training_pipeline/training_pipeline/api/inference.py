import logging
import os
from pathlib import Path
from typing import Tuple, Optional

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training_pipeline import constants, models
from training_pipeline.configs import InferenceConfig
from training_pipeline.data.disaster_tweets_data import DisasterTweetData
from training_pipeline.data import utils

try:
    comet_project_name = os.environ["COMET_PROJECT_NAME"]
except KeyError:
    raise RuntimeError(
        "Veuillez définir la variable environnementale COMET_PROJECT_NAME"
    )

logger = logging.getLogger(name=__name__)


class InferenceAPI:
    """
    Une classe effectuant l'inférence en utilisant un modèle pré-entrainé.
    Args:
        -model_id(str):l'ID du modèle LLM à utiliser.
        -root_dataset_dir(Path):le repertoire racine du dataset
        -test_dataset_file(Path):le chemin du fichier du dataset de test.
        -name(str,optional):le nom de l'API d'inférence.La valeur par défaut est "inference-api".
        -model_cache_dir(Path,optional):le repertoire de cache du modèle.La valeur par défaut est constants.CACHE_DIR.
        -device(str,optional):le périphérique à utiliser pour l'inférence.La valeur par défaut est "cuda:0".
    """

    def __init__(
        self,
        model_id: str,
        root_dataset_dir: Path,
        test_dataset_file: Path,
        name: str = "inference-api",
        model_cache_dir: Path = constants.CACHE_DIR,
        device: str = "cuda:0",
    ):
        self._model_id = model_id
        self._root_dataset_dir = root_dataset_dir
        self._test_dataset_file = test_dataset_file
        self._name = name
        self._model_cache_dir = model_cache_dir
        self._device = device

        self._model, self._tokenizer = self.load_model()
        if self._root_dataset_dir is not None:
            self._dataset = self.load_data()
        else:
            self._dataset = None

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        root_dataset_dir: Path,
        model_cache_dir: Path = constants.CACHE_DIR,
    ):
        """
        Crée une instance de classe InferenceApi à partir d'un object InferenceConfig.
        Args:
            -config(InferenceConfig):l'objet InferenceConfig à utiliser.
            -root_dataset_dir(Path):le repertoire racine du dataset à utiliser.
            -model_cache_dir(Path,optional):le repertoire de la cache du modèle.La valeur par
            défaut est constants.CACHE_DIR.
        Returns:
            InferenceApi:une instance de la classe InferenceApi.
        """

        return cls(
            model_id=config.model["id"],
            root_dataset_dir=root_dataset_dir,
            test_dataset_file=config.dataset["file"],
            model_cache_dir=model_cache_dir,
            device=config.setup.get("device", "cuda:0"),
        )

    def load_data(self) -> Dataset:
        """
        Récupère le dataset des tweets.
        Returns:
            -Dataset:le dataset récupéré.
        """
        logger.info(
            msg=f"Récupération du dataset des tweets à partir de {self._root_dataset_dir=}"
        )

        dataset = DisasterTweetData(
            data_path=self._root_dataset_dir, scope=constants.Scope.INFERENCE
        ).load_data()

        logger.info(msg=f"{len(dataset)} échantillons récupérés pour l'inférence.")

        return dataset

    def load_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Récupère le modèle pour l'inférence.
        Returns:
            -Tuple[AutoModelForSequenceClassification,AutoTokenizer]:un tuple contenant le modèle chargé
            et le tokenizer.
        """
        logger.info(msg=f"Chargement du modèle {self._model_id=}")
        model, tokenizer = models.build_model(
            pretrained_model_name_or_path=self._model_id,
            cache_dir=self._model_cache_dir,
        )

        model.eval()

        return model, tokenizer

    def infer(self, text: str) -> dict:
        """
        Effectue l'inférence en utilisant le modèle fine-tuné récupéré.
        Args:
            -text(str):le tweet à classer.
        Returns:
            -dict:les éléments de la prédiction.
        """

        outputs = models.predict(
            model=self._model,
            tokenizer=self._tokenizer,
            text=text,
            device=self._device,
        )

        return outputs

    def infer_all(self, output_file: Optional[Path] = None) -> None:
        """
        Effectue l'inférence sur l'ensemble d'échantillons récupérés dans dataset.
        Args:
            -output_file(Optional[Path],optional):le fichier de sauvegarde de l'output. la valeur par
            défaut est None.
        """
        assert (
            self._dataset is not None
        ), "Dataset non chargé.Fournir un repertoire dataset au constructeur de la classe:'root_dataset_dir.'"
        outputs = []
        should_save_output = output_file is not None
        for sample in tqdm(self._dataset):
            output = self.infer(text=sample["text"])
            if should_save_output:
                outputs.append(
                    {
                        "true": {
                            "idx": sample["label"],
                            "label": constants.ID2LABEL[sample["label"]],
                        },
                        "prediction": output,
                    }
                )
        if should_save_output:
            utils.write_json(outputs, output_file)
