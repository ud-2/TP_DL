# TP2 : Amélioration des Réseaux de Neurones Profonds

Ce second TP se concentre sur l'optimisation des performances et la lutte contre le surapprentissage (overfitting) via des techniques d'ingénierie avancées.

## Objectifs du Projet
*   **Diagnostic** : Analyse du Biais et de la Variance.
*   **Régularisation** : Implémentation du **Dropout** et de la **Régularisation L2**.
*   **Optimisation** : Comparaison des algorithmes **Adam**, **RMSprop** et **SGD**.
*   **Accélération** : Utilisation de la **Batch Normalization**.

## Structure du Projet
```text
.
├── run_experiments.py  # Script automatisant 6 runs comparatifs
├── requirements.txt    # Dépendances (TensorFlow, MLflow, NumPy)
└── README.md           # Documentation
```

## Installation et Utilisation

### 1. Lancer la suite d'expériences
```bash
python run_experiments.py
```
Le script entraîne automatiquement plusieurs variantes du modèle (Base, Regularized, Batch Norm, et différents optimiseurs).

### 2. Visualisation
```bash
mlflow ui
```
Accédez à `http://localhost:5000` pour comparer les performances.

## Résultats et Analyse (Exécution Réelle)

L'analyse de l'expérience via le graphique **Parallel Coordinates Plot** de MLflow (6 runs) montre :

1.  **Supériorité des optimiseurs adaptatifs** : `Adam` et `RMSprop` atteignent une précision supérieure à **97%** beaucoup plus rapidement que le `SGD_with_momentum` (**95.8%**).
2.  **Effet de la Régularisation** : L'ajout de `L2+Dropout` réduit légèrement la précision sur le set d'entraînement mais stabilise l'écart avec la validation, prouvant une meilleure capacité de généralisation.
3.  **Batch Normalization** : L'activation de la `Batch Norm` a permis une convergence plus stable, même si elle nécessite un monitoring précis des activations pour éviter des oscillations en fin d'entraînement.

### Conclusion Technique
L'optimiseur **Adam** sans régularisation agressive reste le plus performant sur le dataset MNIST. Cependant, pour la robustesse, la combinaison **Adam + Dropout (0.2)** offre le meilleur compromis entre performance et généralisation.

---
**Auteurs** : VUIDE OUENDEU FRANCK JORDAN (21P018)  
**Institution** : ENSPY 5GI  
**Date** : Janvier 2026