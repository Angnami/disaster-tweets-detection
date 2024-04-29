from pathlib import Path
import fire

from beam import App, Image, Runtime, Volume, VolumeType

from training_pipeline import configs


trainig_app = App(
    name="train_disastertweets",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="A10G",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(name="disastertweets_dataset", path="./disastertweets_dataset"),
        Volume(
            name="train_disastertweets_output",
            path="./models",
            volume_type=VolumeType.Persistent,
        ),
        Volume(
            name="model_cache", path="./model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)


@trainig_app.run()
def train(
    config_file: str,
    output_dir: str,
    dataset_dir: str,
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
):
    """
    Entraine un modèle de ML en utilisant le fichier de configuration spécifié et un dataset.
    Args:
        -config_file(str):le chemin du fichier de configuration pour le processus d'entrainement.
        -outout_dir(str):le chemin de sauvegarde du modèle entrainé.
        -dataset_dir(str):le chemin du dataset à utiliser pour l'entrainement.
        -env_file_path(str,optionnel):le chemin du fichier des variables environnementales. .env par défaut.
        -logging_config_path(str,optionnel):le chemin du fichier de configuration. logging.yaml par défaut.
        -model_cache_dir(str,optionnel):le repertoire de cache du modéle. None par défaut.

    """
    from training_pipeline import initialize
    import logging

    # S'assurer d'initialiser les variables environnementales avant d'importer tout autre module
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from training_pipeline import utils
    from training_pipeline.api.training import TrainingAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file = Path(config_file)
    output_dir = Path(output_dir)
    root_dataset_dir = Path(dataset_dir)
    model_cache_dir = Path(model_cache_dir) if model_cache_dir else None

    training_config = configs.TrainingConfig.from_yaml(config_file, output_dir)
    training_api = TrainingAPI.from_config(
        config=training_config,
        root_dataset_dir=root_dataset_dir,
        model_cache_dir=model_cache_dir,
    )
    training_api.train()


if __name__ == "__main__":
    fire.Fire(train)
