from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

from transformers import TrainingArguments

from training_pipeline.data.utils import load_yaml


@dataclass
class TrainingConfig:
    """
    Une classe de confirmation utilisée pour charger et stocher la confirmation d'entrainement.
    Attributes:
        training(TrainingArguments) : arguments d'entrainement du modèle
        model(Dict[str,Any]):dictionnaire contenant les informations du modèle
    """

    training: TrainingArguments

    model: Dict[str, Any]

    @classmethod
    def from_yaml(cls, config_path: Path, output_dir: Path) -> "TrainingConfig":
        """
        Recpuère un fichier de configuration à partir du chemin config_path.
        Args:
            config_path(Path): chemin de récupération du fichier de configuration
            output_dir(Path): chemin d'enregistrement du fichier
        Returns:
            TrainingConfig: objet de configuration d'entrainement du modèle
        """
        config = load_yaml(config_path)

        config["training"] = cls._dict_to_training_arguments(
            training_config=config["training"], output_dir=output_dir
        )
        return cls(**config)

    @classmethod
    def _dict_to_training_arguments(
        cls, training_config: dict, output_dir: Path
    ) -> TrainingArguments:
        """
        Construit un object TrainingArguments à partir d'un dictionnaire de configuration.
        Args:
            - training_config(dict): le dictionnaire contenant la configuration d'entrainement.
            - output_dir(Path): le chemin de sauvegarde de l'output.
        Returns:
            - TrainingArguments: l'objet TrainingArguments
        """
        return TrainingArguments(
            output_dir=str(output_dir),
            logging_dir=str(output_dir / "logs"),
            overwrite_output_dir=training_config["overwrite_output_dir"],
            do_eval=training_config["do_eval"],
            do_train=training_config["do_train"],
            per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
            per_device_train_batch_size=training_config["per_device_train_batch_size"],
            optim=training_config["optim"],
            logging_steps=training_config["logging_steps"],
            learning_rate=training_config["learning_rate"],
            num_train_epochs=training_config["num_train_epochs"],
            lr_scheduler_type=training_config["lr_scheduler_type"],
            evaluation_strategy=training_config["evaluation_strategy"],
            disable_tqdm=training_config["disable_tqdm"],
            report_to=training_config["report_to"],
            save_strategy=training_config["save_strategy"],
            seed=training_config["seed"],
            fp16=training_config["fp16"],
            weight_decay=training_config["weight_decay"],
            load_best_model_at_end=training_config["load_best_model_at_end"],  
            # eval_steps=training_config["eval_steps"],
            # save_steps=training_config["save_steps"],
            #eval_accumulation_steps=training_config["eval_accumulation_steps"],
            save_total_limit=training_config["save_total_limit"],
            metric_for_best_model=training_config["metric_for_best_model"],
            greater_is_better=training_config["greater_is_better"],
            eval_delay=training_config["eval_delay"],
            logging_strategy=training_config["logging_strategy"],
            #warmup_steps=training_config["warmup_steps"]          
        )


@dataclass
class InferenceConfig:
    """
    Une classe représentant la configuration de l'inférence.
    Attributes:
        - model(Dict[str, str]): un dictionnaire contenant la configuration du modèle.
        - setup_config(Dict[str, str]):  un dictionnaire contenant la configuration d'installation.
        - dataset(Dict[str, str]): un dictionnaire contenant la configuration du dataset.
        - ft_model(Dict[str, Any]): un dictionnaire contenant la configuration du modèle fine-tuné.

    """

    model: Dict[str, str]
    setup: Dict[str, str]
    dataset: Dict[str, str]
    ft_model: Dict[str, str]

    @classmethod
    def from_yaml(cls, config_path: Path):
        """
        Télécharge un fichier de configuration à partir du chemin inidiqué.
        """

        config = load_yaml(config_path)

        return cls(**config)
    