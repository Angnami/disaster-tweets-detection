from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from pathlib import Path
import torch


def build_model(
    ft_model_path_or_name: Path,
) -> [AutoModelForSequenceClassification, AutoTokenizer]:
    
    model = AutoModelForSequenceClassification.from_pretrained(ft_model_path_or_name)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=ft_model_path_or_name,
        padding="max_length",
        truncation=True,
        model_max_length=512,
    )

    return model, tokenizer


def predict(
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device:torch.device,
):
    """
    Cette fonction détecte un tweet correspondant à une catastrophe naturelle en utilisant le modèle et le tokenizer.
    Args:
        - model(transformers.PretrainedModel): le modèle à utiliser pour classer le texte (tweet).
        - tokenizer(transformers.PretrainedTokenizer): le tokenizer à utilser pour classer le texte (tweet).
        - text(str): le texte du tweet à classer.
        - device(str,optional): le périphérique à utiliser pclasser le texte (tweet).
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
