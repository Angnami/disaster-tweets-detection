import logging
import fire
from beam import App, Image, Runtime, Volume, VolumeType

logger = logging.getLogger(__name__)

# === Définition l'application Beam ===

disaster_detector = App(
    name="disaster_detector",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="T4",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(
            path="./model_cache", name="model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)

# === Définition d'une fonction de récupération des artefacts ===


def load_detector(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
):
    from inference_pipeline import initialize

    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from inference_pipeline import utils
    from inference_pipeline.disaster_tweet_detector import DisasterTweetsDetector

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)
    detector = DisasterTweetsDetector(model_cache_dir=model_cache_dir)

    return detector


@disaster_detector.rest_api(keep_warm_seconds=300, loader=load_detector)
def run(tweet):
    """
    Exécute le detector via l'API RESTful de Beam.

    Args:
        text (str): Le tweet à classer.

    Returns:
        dict: le résultat de la prédiction.
    """

    response = _run(tweet=tweet)

    return response


def _run(tweet):
    """
    fonction qui appelle le detector et retourne la réponse.

    Args:
        text (str): Le tweet à classer.

    Returns:
        dict: le résultat de la prédiction.
    """

    from inference_pipeline import utils

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    detector = load_detector()

    response = detector.predict(tweet=tweet)

    return response


def run_local(tweet):
    """
    Exécute le detector en local.

    Args:
        text (str): Le tweet à classer.

    Returns:
        dict: le résultat de la prédiction.
    """

    detector = load_detector(model_cache_dir=None)

    response = detector.predict(tweet=tweet)

    return response


if __name__ == "__main__":
    fire.Fire(run_local)
