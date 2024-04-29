import os
import logging
import subprocess
import psutil
import torch


logger = logging.getLogger(__name__)


def log_available_gpu_memory():
    """
    Cette fonction enregistre la mémoire GPU disponible pour chaque périphérique GPU disponible.
    S'il n'existe aucune GPU disponible, elle affiche 'Pas de GPU disponibles'
    Returns:
        None
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info = subprocess.check_output(
                f"nvidia-smi -i {i} --query-gpu=memory.free --format=csv, nounits,noheader",
                shell=True,
            )
            logger.info(msg=f"GPU {i}, mémoire disponible: {memory_info} Mi")
    else:
        logger.info(msg="Il n'y a pas de GPUs disponibles.")


def log_available_ram():
    """
    Cette fonction enregistre la quantité de RAM disponible en gigaoctets ou gigabytes(GB).
    Returns:
        None
    """
    memory_info = psutil.virtual_memory()

    logger.info(
        msg=f"Mémoire RAM disponible: {memory_info.available/(1024.0 ** 3):.2f} GB"
    )


def read_requirements(file_path: str) -> list:
    """
    Cette fonction prend un fichier contenant une liste des dépendances (requirements) et reourne une liste
    extirpée des espaces inutiles.
    Args:
        - file_path: str: chemin du fichier requirements à récupérer
    Returns:
        - requirements : Une liste extirpée des caractères inutiles (espaces)
    """
    with open(file=file_path, mode="r") as file:
        requirements = [line.strip() for line in file if line.strip()]

    return requirements


def log_files_and_subdirs(directory_path: str):
    """
    Cette fonction enregistre tous les fichiers et les sous-repertoires présents dans le repertoire indiqué.
    Args:
        - directory_path (str): Le chemin du repertoire à enregistrer.
    Returns:
        - None
    """
    # Verifier si le repertoire existe
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        for dirpath, dirnames, filenames in os.walk(directory_path):
            logger.info(msg=f"Repertoire: {dirpath}")
            for filename in filenames:
                logger.info(msg=f"Fichier: {os.path.join(dirpath,filename)}")
            for dirname in dirnames:
                logger.info(msg=f"Sous-repertoire: {os.path.join(dirpath, dirname)}")
    else:
        logger.info(msg=f"Le repertoire {directory_path} n'existe pas.")
