# TP1 : De la conception au déploiement de modèles de Deep Learning

Ce projet marque le début du cycle d'ingénierie du Deep Learning. Il couvre l'ensemble du pipeline : de l'entraînement d'un modèle de classification de chiffres (**MNIST**) à sa mise en production via une **API REST** conteneurisée.

## Objectifs du Projet
*   **Modélisation** : Construction d'un réseau dense (Fully-Connected) avec Keras.
*   **Tracking MLOps** : Utilisation de **MLflow** pour le suivi des métriques et la sauvegarde du modèle.
*   **Serving** : Création d'un service de prédiction avec **Flask**.
*   **Industrialisation** : Conteneurisation de l'application avec **Docker**.

## Structure du Projet
```text
.
├── train_model.py      # Entraînement et suivi MLflow
├── app.py              # API Flask pour les prédictions
├── mnist_model.h5      # Modèle entraîné (généré)
├── requirements.txt    # Dépendances (TensorFlow, Flask, MLflow)
├── Dockerfile          # Configuration du conteneur
└── README.md           # Documentation
```

## Installation et Utilisation

### Prérequis
*   Python 3.8+
*   Docker (pour le déploiement)
*   Environnement global `ai_env` actif

### 1. Entraînement
```bash
python train_model.py
```

### 2. Test de l'API (Local)
Lancer le serveur : `python app.py`.
Tester avec un vecteur de 784 zéros :
```bash
python3 -c "import json; print(json.dumps({'image': [0]*784}))" | curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d @-
```

## Résultats Obtenus (Exécution Réelle)

### Performance du Modèle
*   **Précision sur les données de test** : **97.83%**
*   **Loss de test** : 0.0677
*   **Suivi MLflow** : Les paramètres (epochs=5, batch_size=128) et les métriques ont été enregistrés avec succès dans l'interface `mlflow ui`.

### Validation de l'API
L'API a été validée avec succès. Pour une entrée neutre (zéros), le modèle a renvoyé une prédiction de classe **5** avec la distribution des probabilités associée, confirmant que le serveur Flask charge et utilise correctement le fichier `.h5`.

---
**Auteurs** : VUIDE OUENDEU FRANCK JORDAN (21P018)  
**Institution** : ENSPY 5GI  
**Date** : Janvier 2026
