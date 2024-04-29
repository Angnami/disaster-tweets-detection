import logging
from pathlib import Path
from typing import Optional, Tuple
import os 
import comet_ml
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    Trainer,
)

from training_pipeline import constants, metrics, models
from training_pipeline.configs import TrainingConfig
from training_pipeline.data.disaster_tweets_data import DisasterTweetData

logger = logging.getLogger(name=__name__)


class BestModelToModelRegistryCallback(TrainerCallback):
    """
    C'est un callback qui enregistre la meilleure version du modèle dans le registre de modèle comet.ml
    Args:
        - model_id(str): l'ID du modèle à enregistrer dans le registre de modèle.

    """

    def __init__(self, model_id: str):
        self.model_id = model_id

    @property
    def model_name(self) -> str:
        """
        Renvoie le nom du modèle à enregister dans le registre de modèle.
        """
        return f"disaster-tweets-detection/{self.model_id}"

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        C'est un événement appelé à la fin de chaque époque.
        Enregistre la meilleure version du modèle dans le registre comet.ml.

        """
        best_model_checkpoint = state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(
                msg=f"Enregistrement du melleur modèle à partir de {best_model_checkpoint} dans le registre de modèle."
            )
            self.to_model_registry(best_model_checkpoint)
        else:
            logger.warning(
                msg="Pas de meileure version trouvée. L'enregistrement dans le regsitre n'est pas effectué."
            )

    def to_model_registry(self, checkpoint_dir: Path):
        """
        Enregistre un checkpoint donné du model dans le registre Comet.ml
        Args:
            - checkpoint_dir(Path): le chemin du repertoire contenant le checkpoint du modèle.

        """
        
        checkpoint_dir = checkpoint_dir.resolve()

        assert (
            checkpoint_dir.exists()
        ), f"Le repertoire du checkpoint {checkpoint_dir} n'existe pas."

        # Recupérer l'expérience obsolète à partir du contexte
        # global afin d'obtenir la clé d'API et l'ID de l'expérience.
        stale_experiment = comet_ml.get_global_experiment()
        # Reprendre l'expérience en utilisant sa clé API et son ID d'expérience
        experiment = comet_ml.ExistingExperiment(previous_experiment=stale_experiment.get_key(), 
                                                 api_key=os.environ["COMET_API_KEY"])
        if experiment:
            logger.info(
                msg=f"Début de l'enregistrement du checkpoint du modèle @ {self.model_name}"
            )
            experiment.log_model(self.model_name, str(checkpoint_dir))
            experiment.register_model(model_name=self.model_name,registry_name="disater-tweets-detection")
            logger.info(
                msg=f"Fin de l'enregistrement du checkpoint du modèle @ {self.model_name}"
            )
            experiment.end()

class TrainingAPI:
    """
    Une classe pour l'entrainement d'un modèle.
    Args:
        -root_dataset_dir(Path): le repertoire racine du dataset
        -model_id(str): l'identifiant du modèle à utiliser.
        -training_arguments(TrainingArguments): les arguments d'entrainement.
        -name(str,optional):le nom de l'API d'entrainement. Le nom par défaut est "training-api".
        -model_cache_dir(Path, optional): le repertoire de cache du modèle. la valeur par défaut est
        constants.CACHE_DIR
    """

    def __init__(
        self,
        root_dataset_dir: Path,
        model_id: str,
        training_arguments: TrainingArguments,
        name: str = "training-api",
        model_cache_dir: Path = constants.CACHE_DIR,
    ):
        self._root_dataset_dir = root_dataset_dir
        self._model_id = model_id
        self._training_arguments = training_arguments
        self._name = name
        self._model_cache_dir = model_cache_dir

        self._model, self._tokenizer = self.load_model()
        self._training_dataset, self._validation_dataset = self.load_data()

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        root_dataset_dir: Path,
        model_cache_dir: Optional[Path] = None,
    ):
        """
        Crée une instance de TrainingApi à partir d'un objet TrainingConfig.
        Args:
            -config(TrainingConfig):la configuration d'entrainement.
            -root_dataset_dir(Path): le repertoire racine du dataset.
            -model_cache_dir(Path,optional): le repertoire de cache du modèle. La valeur par défaut est None.
        Returns:
            - TrainingAPI:une instance de TrainingAPI.
        """
        return cls(
            root_dataset_dir=root_dataset_dir,
            model_id=config.model["id"],
            training_arguments=config.training,
            model_cache_dir=model_cache_dir,
        )

    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Charge les datasets d'entrainement et de validation.
        Returns:
            -Tuple[Dataset, Dataset]: un tuple contenant les datasets d'entrainement et de validation.
        """
        logger.info(msg=f"Chargement des Datasets à partir {self._root_dataset_dir=}")
        dataset = DisasterTweetData(
            data_path=self._root_dataset_dir, scope=constants.Scope.TRAINING
        ).load_data()
        training_dataset = dataset["train"].map(
            self._tokenizer,
            input_columns=["text"],
            remove_columns=["text"],
            batched=True,
            batch_size=None,
        )
        training_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        validation_dataset = dataset["test"].map(
            self._tokenizer,
            input_columns=["text"],
            remove_columns=["text"],
            batched=True,
            batch_size=None,
        )
        validation_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        logger.info(msg=f"Training dataset size: {len(training_dataset)}")
        logger.info(msg=f"Validation dataset size:{len(validation_dataset)}")

        return training_dataset, validation_dataset

    def load_model(self) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
        """
        Recupère le modèle.
        Returns:
            -Tuple[AutoModelForSequenceClassification, AutoTokenizer]:un tuple contenant le modèle
            et le tokenizer.
        """
        logger.info(msg=f"Chargement du modèle en utilisant {self._model_id=}")
        model, tokenizer = models.build_model(
            pretrained_model_name_or_path=self._model_id,
            cache_dir=self._model_cache_dir,
        )

        return model, tokenizer

    def train(self) -> Trainer:
        """
        Entraine le modèle.
        Returns:
            -Trainer:le modèle entrainé.
        """
        logger.info(msg="Entrainement du modèle...")

        trainer = Trainer(
            model=self._model,
            train_dataset=self._training_dataset,
            eval_dataset=self._validation_dataset,
            args=self._training_arguments,
            compute_metrics=metrics.compute_metrics,
            callbacks=[BestModelToModelRegistryCallback(model_id=self._model_id)],
            tokenizer=self._tokenizer,
        )

        trainer.train()
        return trainer
