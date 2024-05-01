<div align="center">
    <h2>Disaster-tweets-detection </h2>
    <h1>Developper un modèle de Deep Learning pour la détection des tweets liés aux catastrophes naturelles</h1>
    <i>Par <a href="https://github.com/Angnami/">Goudja Mahamat</a></i>
</div>

# Description

Ce projet est essentiellement basé sur [le travail](https://github.com/iusztinpaul/hands-on-llms) réalisé par <i> <a href="https://github.com/iusztinpaul">Paul Iusztin</a>, <a href="https://github.com/Paulescu">Pau Labarta Bajo</a> et <a href="https://github.com/Joywalker">Alexandru Razvant</a></i>. 
Il correspond à la création d'un modèle de Deep Learning pour la détection des tweets liés aux catastrophes naturelles à partir des données de kaggle. 

# Démarche
Le projet est réalisé en suivant deux étapes principales à savoir:  

- L'entrainement (Préparation des données incluse);
- L'entrainement.

## Pipeline d'entrainement/Fine-Tuning   

Ce module permet de : 
- Charger ces [données](https://www.kaggle.com/competitions/nlp-getting-started/data) de kaggle;
- Fine-tuner un [modèle](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) de base HugginFace;
- Enregistrer les experiences dans l'outil de suivi des expériences de Comet ML;
- Sauvegarder le meilleur modèle dans le registre de modèles de Comet ML .   

## Pipeline d'inférence 

Ce module permet de : 
- Charger le modèle fine-tuné et sauvegardé dans le registre de modèle de Comet ML;
- Prendre un tweet (saisi par l'utilsateur ou choisi aléatoiré dans les données non connues par le modèle lors de son entrainement),  le pré-traiter;
- Afficher les résultats de la prédiction en disant si le tweet correspond à une catastrophe naturelle ou pas;
- Créer une interface Streamlit permettant aux utilisateurs d'interagir avec le modèle.   