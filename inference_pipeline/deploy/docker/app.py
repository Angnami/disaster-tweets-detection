import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing import clean_tweet
import models
import torch
# Utilisation de GPU s'il existe
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Interface Streamlit
# Configuration de la page 
st.set_page_config(page_title="Disaster-Tweets-Dector",layout="wide", page_icon="random")
# Titre
st.title("Disaster Tweets Detection Application")
# Fonction de récupération du modèle
@st.cache_resource
def load_model():
    ft_model_path_or_name = Path("./model/angnami/disaster-tweets-detection-distilbert:1.0.0")
    # Définition du détecteur
    model, tokenizer = models.build_model(ft_model_path_or_name=ft_model_path_or_name)
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
