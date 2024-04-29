import logging
import logging.config
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)


def initialize(logging_config_path: str = "logging.yaml", env_file_path: str = ".env"):
    """
    Initialise le logger et les variables d'environnement.
    Args:
        -logging_config_path(str):le chemin du fichier de configuration de logging. logging.yaml par défaut.
        -env_file_path(str):le chemin du fichier de configuration des variables d'environnement. .env par défaut.
    """
    logger.info(msg="Initialisation du logger...")

    try:
        initialize_logger(config_path=logging_config_path)
    except FileNotFoundError:
        logger.warning(
            msg=f"Fichier de configuration non trouvé à {logging_config_path}. Définition du niveau de logging à INFO"
        )
        logging.basicConfig(level=logging.INFO)

    logger.info(msg="Initialisation des variables d'environnement...")
    if env_file_path is None:
        env_file_path = find_dotenv(raise_error_if_not_found=False, usecwd=False)

    if env_file_path is not None:
        logger.info(
            msg=f"Chargement des variables d'environnement à partir de {env_file_path}"
        )
        load_dotenv(dotenv_path=env_file_path, verbose=True, override=True)


def initialize_logger(
    config_path: str = "logging.yaml", log_dir_name: str = "logs"
) -> logging.Logger:
    """
    Initialise un logger à partir d'un fichier de configuration YAML.
    Args:
        -config_path(str):le chemin du fichier de configuration de logging. logging.yaml par défaut.
        -log_dir_name(str):le fichier de sauvegarde des loggings.
    """
    # Créer un repertoire logs
    config_path_parent = Path(config_path).parent
    logs_dir = config_path_parent / log_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "rt") as f:
        config = yaml.safe_load(f.read())

    # S'assurer le fichier de configuration existant focntionne toujours
    config["disable_existing_loggers"] = False

    logging.config.dictConfig(config=config)
