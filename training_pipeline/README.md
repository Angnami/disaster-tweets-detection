# Pipeline d'entrainement/Fine-Tuning   

Ce module permet de : 
- Charger ces [données](https://www.kaggle.com/competitions/nlp-getting-started/data) de kaggle;
- Fine-tuner un [modèle](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) de base HugginFace;
- Enregistrer les experiences dans l'outil de suivi des expériences de Comet ML;
- Sauvegarder le meilleur modèle dans le registre de modèles de Comet ML .   

## Table des matières  

- [1. Installation](#1-install)
    - [1.1 Dépendances](#1.1-dependancies)
    - [1.2 Beam](#1.2-beam)
- [2. Utilisation](#2-usage)
    - [2.1 Entrainement](#2.1-train)
    - [2.2 Inférence](#2.2-inference)
    - [2.3 Linting & formatage](#2.3-linting--formatting)  

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

## 2.1. Entrainement  
`Effectuer l'entrainement, suivre l'expérimentation et enregistrer le modèle dans Comet ML`

### Local

For debugging or to test that everything is working fine, run the following to train the model on a lower number of samples:
```
make dev_train_local
```

For training on the production configuration, run the following:
```shell
make train_local
```

### Sur Beam

Pour l'entrainement sur Beam, exécuter:
```shell
make train_beam
```

## 2.2. Inférence
`Effectuer l'inférence`


### sur Beam

Pour effectuer l'inférence, exécuter::
```shell
make infer_beam
```

## 2.3. Linting & Formatage

**Vérifier** le code pour détecter les problèmes de **linting**:
```shell
make lint_check
```

**Corriger** le code des problèmes de **linting** issues:
```shell
make lint_fix
```
