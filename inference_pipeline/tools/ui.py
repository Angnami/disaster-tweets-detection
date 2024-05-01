import argparse
import logging

import streamlit as st

import pandas as pd
import numpy as np

from inference_pipeline.data.utils import clean_tweet

logger = logging.getLogger(__name__)

# Fonction d'analyse des variables d'environnement
def parseargs() -> argparse.Namespace:
    """
    Analyse les arguments de la ligne de commande pour faire la prédiction.

    Returns:
        argparse.Namespace: Un objet contenant les arguments analysés.
    """

    parser = argparse.ArgumentParser(description="Disaster Tweets Detector Model")

    parser.add_argument(
        "--env-file-path",
        type=str,
        default=".env",
        help="Chemin du fichier des variables d'environnement",
    )

    parser.add_argument(
        "--logging-config-path",
        type=str,
        default="logging.yaml",
        help="Chemin du fichier de configuration",
    )

    parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="./model_cache",
        help="Chemin de la cache du modèle",
    )

    return parser.parse_args()


args = parseargs()


# Récupération des artefacts
def load_detector(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
):
    """Cette fonction charge les variables d'environnement, la configuration des loggings et définit le détecteur.
    Args:
        env_file_path: Fichier des variables d'environnement
        logging_config_path: Fichier de configuration des loggings
        model_cache_dir: l'addresse de cache du modèle
    """
    from inference_pipeline import initialize
    # Charger les variables d'environnement et la configuration des loggings
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)
    
    from inference_pipeline import utils
    from inference_pipeline.disaster_tweet_detector import DisasterTweetsDetector

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)
    # Définition du détecteur
    detector = DisasterTweetsDetector(model_cache_dir=None)

    return detector

# Définition du détecteur
detector = load_detector(
    env_file_path=args.env_file_path, logging_config_path=args.logging_config_path
)

# Interface Streamlit
# Configuration de la page
st.set_page_config(page_title="Disaster-Tweets-Dector",layout="wide", page_icon="random")
# Titre
st.title("Disaster Tweets Detection Application")

# st.subheader("@Goudja")

# Fonction pour télécharger des exemples de tweets et les mettre en cache
@st.cache_data
def load_data():
    df = pd.read_csv(filepath_or_buffer="./dataset/test.csv", usecols=["text"])
    return df
# Les données
data = load_data()

# Choisir l'option d'usage du modèle
option = st.selectbox(
    label="Choisissez une option d'utilisation du modèle",
    options=["mon propre tweet", "un tweet existant"],
    index=1,
    key="selection"
)

    
if option == "mon propre tweet":
    with st.form(key="My own tweet",clear_on_submit=True):
        #Ecrire un tweet
        tweet = st.text_area(label="Veuillez écrire votre tweet",key="tweet")
        
        if tweet:
            # Nettoyer le tweet écrit
            preprocessed_tweet = clean_tweet(tweet=tweet)
            
            # Afficher le tweet néttoyé
            st.text(preprocessed_tweet, help="Version pré-traitée de votre tweet")
            
            # Faire la prédiction sur le tweet
            prediction = detector.predict(preprocessed_tweet)
            
            # Afficher les résultats de la prédiction
            st.write(prediction)
        st.form_submit_button(label="Afficher les résultats")
else:
    # Choisir un nombre aléatoire
    random_id = np.random.randint(low=0, high=data.shape[0])

    # Récupérer le tweet correspondant à l'index tiré aléatoirement
    tweet = data.iloc[random_id].values.tolist()[0]

    # Néttoyer le tweet
    preprocessed_tweet = clean_tweet(tweet=str(tweet))
    # Afficher le tweet
    st.text(tweet,help="Exemple de tweet choisi aléatoirement")
    #Afficher la version nétoyée du tweet
    st.text(preprocessed_tweet,help="Version pré-traitée de ce tweet")

    # Faire la prédiction de la classe du tweet
    prediction = detector.predict(preprocessed_tweet)
    # Afficher les résultats de la prédiction
    st.write(prediction)
    
    # Pour afficher un autre exemple de tweet
    get_random_tweet = st.button(label="Afficher un autre tweet")
