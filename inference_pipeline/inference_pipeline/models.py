import os
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import logging
from pathlib import Path
from typing import Optional
from comet_ml import API
import torch
from inference_pipeline import constants

logger = logging.getLogger(__name__)


def build_model(
    ft_model_path_or_name: str,
    model_cache_dir: Optional[Path] = None,
) -> [AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Cette fonction recupère le modèle fine-tuné.
    1. charge les artefacts du modèle s'ils sont disponibles en cache
    2. télécharge le modèle à partir du registre de modèles de COMET ML s'il n'est pas disponible en cache

    Args:
        ft_model_path_or_name(str,optionnel):le nom ou le chemin du modèle fine-tuné
        cache-dir(Path):le repertoire de la cache où sera enregistré le modèle.
    Returns:
        [AutoModelForSequenceClassification, AutoTokenizer]:un tuple contenant le modèle construit et le tokenizer
    """
    is_model_name = not os.path.isdir(ft_model_path_or_name)
    if is_model_name:
        logger.info(
            f"Téléchargement de {ft_model_path_or_name} à partir du registre de modèles de COMET ML."
        )
        ft_model_path_or_name = download_from_model_registry(
            model_id=ft_model_path_or_name,
            cache_dir=model_cache_dir,
        )
    logger.info(f"Chargement du modèle fine-tuné dans: {ft_model_path_or_name}")
    model = AutoModelForSequenceClassification.from_pretrained(ft_model_path_or_name)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=ft_model_path_or_name,
        padding="max_length",
        truncation=True,
        cache_dir=str(model_cache_dir),
        model_max_length=512,
    )

    return model, tokenizer


def download_from_model_registry(
    model_id: str, cache_dir: Optional[Path] = None
) -> Path:
    """
    Cette fonction télécharge un modèle à partir du registre de modèle de Comet ML.
    Args:
        - model_id(str): ID du modèle à télécharger sous la forme "workspace/model_name:version"
        - cache_dir(Path): le repertoire de la cache du modèle téléchargé
    Returns:
        - Le chemin du modèle téléchargé
    """
    if cache_dir is None:
        output_folder = constants.CACHE_DIR / "models" / model_id
    else:
        output_folder = Path(cache_dir)/model_id
    already_downloaded = output_folder.exists()
    if not already_downloaded:
        workspace, model_id = model_id.split("/")
        model_name, version = model_id.split(":")

        api = API()
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(version=version, output_folder=output_folder)
    else:
        logger.info(msg=f"Le modèle {model_id} est déjà téléchargé à {output_folder}")

    logger.info(
        msg=f"Le modèle {model_id} est téléchargé depuis le registre dans {output_folder}"
    )

    return output_folder


def predict(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str = "cuda:0",
):
    """
    Cette fonction détecte un tweet correspondant à une catastrophe naturelle en utilisant le modèle et le tokenizer.
    Args:
        - model(transformers.PretrainedModel): le modèle à utiliser pour classer le texte (tweet).
        - tokenizer(transformers.PretrainedTokenizer): le tokenizer à utilser pour classer le texte (tweet).
        - text(str): le texte du tweet à classer.
        - device(str,optional): le périphérique à utiliser pclasser le texte (tweet). La valeur par défaut est cuda:0.
    Returns:
        - le résultat fourni par le modèle.
    """
    inputs = tokenizer(
        text=text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    ).to(device)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs[0].softmax(1).detach().cpu().numpy()
    pred_idx = probs.argmax()
    pred_label = model.config.id2label[pred_idx.item()]

    return {
        "probs": probs.squeeze(axis=0).tolist(),
        "idx": int(pred_idx),
        "label": pred_label,
    }
