import logging
from pathlib import Path
import streamlit as st
import os
import pandas as pd
import numpy as np
from preprocessing import clean_tweet
from constants import FINE_TUNED_MODEL_CHKPT
import models
import torch

# Utilisation de GPU s'il existe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


# Interface Streamlit
# Configuration de la page 
st.set_page_config(page_title="Disaster-Tweets-Dector",layout="wide", page_icon="random")
# Titre
st.title("Disaster Tweets Detection Application")
# Récupération des variables d'environnement
os.environ["COMET_API_KEY"] = st.secrets["comet_credentials"]["COMET_API_KEY"]
os.environ["COMET_WORKSPACE"] = st.secrets["comet_credentials"]["COMET_WORKSPACE"]
os.environ["COMET_PROJECT_NAME"] = st.secrets["comet_credentials"]["COMET_PROJECT_NAME"]
# Fonction de récupération du modèle
@st.cache_resource
def load_model(
    #env_file_path:str=".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_outputs",
    ft_model_path_or_name: str = FINE_TUNED_MODEL_CHKPT
):
    model_cache_dir = Path(model_cache_dir)
    from inference_pipeline import initialize_logger
    # Charger la configuration des loggings
    try:
        initialize_logger(config_path=logging_config_path)
    except FileNotFoundError:
        logger.warning(
            msg=f"Fichier de configuration non trouvé à {logging_config_path}. Définition du niveau de logging à INFO"
        )
        logging.basicConfig(level=logging.INFO)
    from inference_pipeline import utils

    # Vérification des ressources disponibles
    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)
    # Définition du détecteur
    model, tokenizer = models.build_model(
            model_cache_dir=model_cache_dir, ft_model_path_or_name=ft_model_path_or_name
        )
    return model, tokenizer
# Fonction de prédiction
def make_prediction(tweet:str):
    
    return models.predict(model=model, tokenizer=tokenizer, text=tweet, device=device)

# Chargement du modèle
model, tokenizer = load_model()

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
    with st.form(key="My own tweet"):
        #Ecrire un tweet
        tweet = st.text_area(label="Veuillez écrire votre tweet",key="tweet")
        
        if tweet:
            # Nettoyer le tweet écrit
            preprocessed_tweet = clean_tweet(tweet=tweet)
            
            # Afficher le tweet néttoyé
            st.text(preprocessed_tweet, help="Version pré-traitée de votre tweet")
            # Faire la prédiction sur le tweet
            prediction = make_prediction(preprocessed_tweet)
            
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
    prediction = make_prediction(preprocessed_tweet)
    # Afficher les résultats de la prédiction
    st.write(prediction)
    
    # Pour afficher un autre exemple de tweet
    get_random_tweet = st.button(label="Afficher un autre tweet")