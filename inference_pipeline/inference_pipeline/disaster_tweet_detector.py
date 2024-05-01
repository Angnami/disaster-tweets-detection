from inference_pipeline.constants import FINE_TUNED_MODEL_CHKPT
from inference_pipeline import models
from inference_pipeline.data import utils
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DisasterTweetsDetector:
    """
    Cette classe permet de récupérer le modèle enregistré dans le registre de modèles,
    pré-traiter le tweet à classer et renvoie la classe correspondante.
    """

    def __init__(
        self,
        model_cache_dir: str,
        ft_model_path_or_name: str = FINE_TUNED_MODEL_CHKPT,
    ):
        self.model, self.tokenizer = models.build_model(
            model_cache_dir=model_cache_dir, ft_model_path_or_name=ft_model_path_or_name
        )

    def _clean_tweet(self, tweet):
        """Cette méthode permet d'appliquer les différentes étapes de preprocessing au tweet à classer"""
        return utils.clean_tweet(tweet)

    def predict(self, tweet):
        return models.predict(
            model=self.model,
            tokenizer=self.tokenizer,
            text=self._clean_tweet(tweet),
            device=device,
        )
