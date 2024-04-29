from pathlib import Path
import fire
from beam import App, Image, Runtime, Volume, VolumeType, Output
from training_pipeline import configs

inference_app = App(
    name="inference_disastertweets",
    runtime=Runtime(
        cpu=4,
        memory="64Gi",
        gpu="A10G",
        image=Image(python_version="python3.10", python_packages="requirements.txt"),
    ),
    volumes=[
        Volume(name="disastertweets_dataset", path="./disastertweets_dataset"),
        Volume(
            name="model_cache", path="./model_cache", volume_type=VolumeType.Persistent
        ),
    ],
)


@inference_app.task_queue(
    outputs=[Output(path="output-inference/output-inference-api.json")]
)
def infer(
    config_file: str,
    dataset_dir: str,
    output_dir: str = "output-inference",
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = None,
):
    """
    Effectue l'inférence sur un dataset en utilisant un modèle entrainé.
    Args:
        -config_file(str):le chemin du fichier de configuration d'inférence
        -outout_dir(str):le chemin de sauvegarde. "output-inference" par défaut.
        -dataset_dir(str):le chemin du dataset à utiliser pour l'entrainement.
        -env_file_path(str,optionnel):le chemin du fichier des variables environnementales. .env par défaut.
        -logging_config_path(str,optionnel):le chemin du fichier de configuration. logging.yaml par défaut.
        -model_cache_dir(str,optionnel):le repertoire de cache du modéle. None par défaut.
    """
    import logging

    from training_pipeline import initialize

    # S'assurer d'initialiser les variables environnementales avant d'importer tout autre module
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from training_pipeline import utils
    from training_pipeline.api.inference import InferenceAPI

    logger = logging.getLogger(__name__)

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    config_file = Path(config_file)
    root_dataset_dir = Path(dataset_dir)
    model_cache_dir = Path(model_cache_dir) if model_cache_dir else None
    inference_output_dir = Path(output_dir)
    inference_output_dir.mkdir(exist_ok=True, parents=True)

    inference_config = configs.InferenceConfig.from_yaml(config_file)
    inference_api = InferenceAPI.from_config(
        config=inference_config,
        root_dataset_dir=root_dataset_dir,
        model_cache_dir=model_cache_dir,
    )
    inference_api.infer_all(
        output_file=inference_output_dir / "output-inference-api.json"
    )


if __name__ == "__main__":
    fire.Fire(infer)
