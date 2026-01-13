# TP1 : De la conception au déploiement de modèles de Deep Learning

Ce premier projet de Travaux Pratiques pose les jalons du cycle de vie d'un modèle de Deep Learning (DL Engineering). L'objectif est de passer de la théorie (réseaux de neurones) à la pratique industrielle en incluant le suivi, l'empaquetage et le déploiement.

## Objectifs du Projet

*   **Fondations du Deep Learning** : Construction, entraînement et évaluation d'un réseau de neurones *fully-connected* (dense) sur le jeu de données **MNIST**.
*   **Versionnement & Collaboration** : Apprentissage des bonnes pratiques avec **Git** et **GitHub/GitLab**.
*   **Suivi d'Expériences** : Introduction à **MLflow** pour enregistrer les paramètres (époques, taux de dropout) et les métriques de performance.
*   **Déploiement (Serving)** : Création d'une API Web avec le framework **Flask** pour permettre des prédictions en temps réel via des requêtes HTTP.
*   **Conteneurisation** : Utilisation de **Docker** pour encapsuler l'application et ses dépendances dans un conteneur isolé et portable.

## Structure du Projet

```text
.
├── train_model.py      # Script d'entraînement, évaluation et suivi MLflow
├── app.py              # Application Flask servant le modèle via une API REST
├── requirements.txt    # Liste des dépendances (tensorflow, flask, mlflow, numpy)
├── Dockerfile          # Fichier de configuration pour l'image Docker
├── mnist_model.h5      # Modèle entraîné sauvegardé (généré après entraînement)
└── README.md           # Documentation du projet
```

## Prérequis

*   Python 3.8 ou plus.
*   **Git** installé sur votre machine.
*   **Docker** (nécessaire pour la partie 2.3 sur la conteneurisation).
*   Un compte GitHub ou GitLab.

## Installation

1.  **Cloner le dépôt et se placer sur la branche correspondante :**
    ```bash
    git clone https://github.com/ud-2/TP_DL.git
    cd TP_DL
    git checkout tp1
    ```

2.  **Créer et activer un environnement virtuel :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Entraînement et suivi MLflow
Exécutez le script pour entraîner le modèle et enregistrer les données dans MLflow :
```bash
python train_model.py
```
Pour visualiser l'interface de suivi : `mlflow ui` puis rendez-vous sur `http://localhost:5000`.

### 2. Lancement de l'API Flask (Local)
Pour tester l'API localement sans Docker :
```bash
python app.py
```
L'API sera disponible sur `http://localhost:5000/predict`.

### 3. Déploiement avec Docker
1.  **Construire l'image :**
    ```bash
    docker build -t mnist-api .
    ```
2.  **Lancer le conteneur :**
    ```bash
    docker run -p 5000:5000 mnist-api
    ```

## Concepts Clés abordés

*   **Descente de Gradient Stochastique (SGD)** : Pourquoi elle est préférée au gradient classique pour les grands jeux de données.
*   **Rétropropagation du gradient** : Mécanisme de mise à jour des poids du réseau.
*   **Vectorisation & Batching** : Optimisation des calculs matriciels.
*   **Cycle de vie (Pipeline)** : Développement $\to$ Tracking $\to$ Packaging $\to$ Production.
*   **Isolation logicielle** : Intérêt de Docker pour éviter les conflits de versions entre environnements.