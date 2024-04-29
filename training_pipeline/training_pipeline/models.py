from comet_ml import API
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
import torch

from training_pipeline import constants

logger = logging.getLogger(__name__)


def build_model(
    pretrained_model_name_or_path: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    cache_dir: Optional[Path] = None,
    ft_model_path_or_name:Optional[str]=None
) -> [AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Cette fonction recupère un modèle HF.
    1. Télécharge le modèle "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    2. Charge et configure le tokenizer de "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

    Args:
        pretrained_model_name_or_path(str): le nom ou le chemin du modèle pre-entrainé de HuggingFace
        ft_model_path_or_name(str,optionnel):le nom ou le chemin du modèle fine-tuné
        cache-dir(Path):le repertoire de la cache où sera enregistré le modèle.
    Returns:
        [AutoModelForSequenceClassification, AutoTokenizer]:un tuple contenant le modèle construit et le tokenizer
    """
    if ft_model_path_or_name:
        is_model_name = not os.path.isdir(ft_model_path_or_name)
        if is_model_name:
            logger.info(
                f"Téléchargement de {ft_model_path_or_name} à partir du registre de modèles de COMET ML."
            )
            ft_model_path_or_name = download_from_model_registry(
                model_id=ft_model_path_or_name,
                cache_dir=cache_dir,
            )
        ft_base_model = ft_model_path_or_name.split('/')[1].split(':')[0]
        assert (
            pretrained_model_name_or_path == ft_base_model
        ), f"Modèle fine-tuné sur un modèle de base différent que celui demandé: \
        {pretrained_model_name_or_path} != {ft_base_model}"

        logger.info(f"Chargement du modèle fine-tuné: {ft_model_path_or_name}")
        model = AutoModelForSequenceClassification.from_pretrained(ft_model_path_or_name)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            trust_remote_code=False,
            cache_dir=str(cache_dir) if cache_dir else None,
            num_labels=constants.NUM_LABELS,
            id2label=constants.ID2LABEL,
            label2id=constants.LABEL2ID,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        padding="max_length",
        truncation=True,
        cache_dir=str(cache_dir) if cache_dir else None,
        model_max_length=512,
    )

    return model, tokenizer


def download_from_model_registry(model_id: str, cache_dir: Optional[Path]):
    """
    Cette fonction télécharge un modèle à partir du registre de modèle de Comet ML.
    Args:
        - model_id(str): ID du modèle à télécharger sous la forme "workspace/model_name:version"
        - cache_dir(Path): le repertoire de la cache du modèle téléchargé
    Returns:
        - Le chemin du modèle téléchargé
    """
    if cache_dir is None:
        cache_dir = constants.CACHE_DIR
    output_folder = cache_dir / "models" / model_id
    already_downloaded = output_folder.exists()
    if not already_downloaded:
        workspace, model_id = model_id.split("/")
        model_name, version = model_id.split(":")

        api = API()
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(version=version, output_folder=output_folder, expand=True)
    else:
        logger.info(msg=f"Le modèle {model_id} est déjà téléchargé à {output_folder}")

    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"Il doit avoir un seul repertoire dans le dossier du modèle. Vérifier le modèle téléchargé dans {output_folder}"
        )
    logger.info(
        msg=f"Le modèle {model_id} est téléchargé depuis le registre dans {model_dir}"
    )

    return model_dir


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
        "idx": pred_idx,
        "label": pred_label,
    }
