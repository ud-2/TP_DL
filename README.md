# TP2 : Amélioration des Réseaux de Neurones Profonds

Ce projet se concentre sur les techniques avancées pour transformer un modèle de base en un modèle performant et robuste en luttant contre le surapprentissage.

## Objectifs du Projet
*   **Diagnostic** : Analyse du Biais et de la Variance via le découpage des données (Train/Val/Test).
*   **Régularisation** : Implémentation de la régularisation **L2 (Weight Decay)** et du **Dropout**.
*   **Optimisation** : Comparaison des algorithmes **Adam**, **RMSprop** et **SGD avec Momentum**.
*   **Normalisation** : Utilisation de la **Batch Normalization** pour stabiliser et accélérer l'apprentissage.

## Structure du Projet
```text
.
├── run_experiments.py  # Script automatisé lançant plusieurs runs MLflow
├── requirements.txt
└── README.md
```

## Utilisation
Lancer toutes les expériences de comparaison :
```bash
python run_experiments.py
```
Pour visualiser les courbes de convergence et comparer l'impact de la Batch Normalization ou des optimiseurs :
```bash
mlflow ui
```

## Concepts abordés
*   Surapprentissage (Overfitting) vs Sous-apprentissage (Underfitting).
*   Taux d'apprentissage adaptatif.
*   Stabilisation des activations internes (Internal Covariate Shift).