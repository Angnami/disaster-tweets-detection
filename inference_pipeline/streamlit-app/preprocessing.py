import regex as re
import yaml
from pathlib import Path
import json
from typing import List, Union

from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    remove_punctuation,
)


def clean_tweet(tweet: str) -> str:
    """Cette fonction retourne la version transformée d'un tweet.
    Les transformations effectuées sont:
    1. Suppression des urls
    2. Suppression des emails
    3. Suppression des usernames
    4. Suppression des chiffres
    5. Suppression des paranthèses
    6. Suppression des caractères non ascii
    7. Conversion en miniscule
    8. Suppression des emojis
    9. Suppression des tirets
    10. Suppression des puces d'énumération
    11. Suppression des ponctuations
    12. Suppression des espaces inutiles
    13. Regroupement des paragraphes rompus
    14. Suppression des caractères spéciaux
    15. Suppression des underscores
    Args:
        tweet: str
    Returns:
        la version transformée du tweet
    """
    # 1
    tweet = re.sub(r"(https?:\S+)", "", tweet).strip()
    # 2
    tweet = re.sub(r"[\w-_\.]+@[\w-_\.]+", "", tweet).strip()
    # 3
    tweet = re.sub(r"@[\w-_\.]+", "", tweet).strip()
    # 4
    tweet = re.sub(r"\d+", "", tweet).strip()
    # 5
    tweet = re.sub(r"\[.*?\]", "", tweet)
    # 7,9, 10,  12
    tweet = clean(
        tweet, extra_whitespace=True, dashes=True, bullets=True, lowercase=True
    )
    # 6, 8
    tweet = clean_non_ascii_chars(text=tweet)
    # 13
    tweet = group_broken_paragraphs(tweet)
    # 11
    tweet = remove_punctuation(tweet)
    # 14
    tweet = re.sub(r"[^a-zA-z0-9.,!?/:;\"\'\s]", "", tweet)
    # 15
    tweet = re.sub(r"[_]+", "", tweet)

    return tweet


def load_yaml(path: Path) -> dict:
    """Cette fonction recupère un fichier YAML et renvoie son contenu sous
    forme de dictionnaire.
    Args:
        path(Path): le chemin de récupération du fichier YAML.
    Returns:
        dict: le contenu du fichier YAML sous forme de dictionnaire.
    """
    with path.open("r") as file:
        config = yaml.safe_load(file)

    return config


def write_json(data: Union[dict, List[dict]], path: Path) -> None:
    """
    Enregistre un dictionnaire ou une liste de dictionnaires en  fichier json.
    Args:
        - data(Union[dict, List[dict]]): les données à enregistrer.
        - path(Path): le chemin du fichier à enregister.
    Returns:
        - None
    """

    with path.open("w") as f:
        json.dump(obj=data, fp=f, indent=4, default=str)
