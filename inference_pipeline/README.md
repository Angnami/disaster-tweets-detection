# Pipeline d'inférence 

Ce module permet de : 
- Charger le modèle fine-tuné et sauvegardé dans le registre de modèle de Comet ML;
- Prendre un tweet (saisi par l'utilsateur ou choisi aléatoiré dans les données non connues par le modèle lors de son entrainement), le pré-traiter;
- Afficher les résultats de la prédiction en disant si le tweet correspond à une catastrophe naturelle ou pas;
- Créer une interface Streamlit permettant aux utilisateurs d'interagir avec le modèle.   

## Table des matières  

- [1. Installation](#1-install)
    - [1.1 Dépendances](#1.1-dependancies)
    - [1.2 Beam](#1.2-beam)
- [2. Utilisation](#2-usage)
    - [2.1 En local](#2.1-local)
    - [2.2 Déploier sur Beam comme une API RESTful](#2.2-deploy-to-beam)
    - [2.3. Streamlit UI](#2-3-streamlit-ui)
    - [2.4 Linting & formatage](#2.3-linting--formatting)  

# 1. Installation

## 1.1. Dépendances

Dépendances principales à installer soi-même:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3

Installation des autres dépendances en exécutant:
```shell
make install
```

Lors du développement, exécuter:
```shell
make install_dev
```

Préparation des informations d'identification:
```shell
cp .env.example .env
```  

# 2. Utilisation

## 2.1. En local  
Exécuter le modèle en local avec un tweet prédéfini:
```shell
make run
```


## 2.2. Déploier sur Beam comme une API RESTful
```shell
make deploy_beam
```

## 2.3. Streamlit UI
```shell
make run_ui
```

## 2.4. Linting & Formatage

**Vérifier** le code pour détecter les problèmes de **linting**:
```shell
make lint_check
```

**Corriger** le code des problèmes de **linting**:
```shell
make lint_fix
```
