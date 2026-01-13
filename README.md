# TP2 : Amélioration des Réseaux de Neurones Profonds

Ce projet constitue le second volet des travaux pratiques et se concentre sur les techniques d'ingénierie avancées visant à optimiser les performances des modèles, à accélérer leur convergence et à garantir leur robustesse face au surapprentissage.

## Objectifs du Projet

*   **Diagnostic de Performance** : Apprendre à identifier les problèmes de **haut biais** (underfitting) et de **haute variance** (overfitting).
*   **Partitionnement des données** : Mise en place rigoureuse des ensembles d'entraînement, de validation (dev set) et de test.
*   **Régularisation** : Implémentation des techniques de régularisation **L2 (Weight Decay)** et de **Dropout** pour stabiliser les poids du réseau.
*   **Normalisation** : Intégration de la **Batch Normalization** pour stabiliser les activations internes et accélérer l'entraînement.
*   **Optimisation Avancée** : Comparaison et évaluation des algorithmes d'optimisation (Momentum, RMSprop, et **Adam**).
*   **Suivi d'Expériences** : Utilisation systématique de **MLflow** pour comparer les différentes configurations de modèles.

## Structure du Projet

```text
.
├── run_experiments.py  # Script principal automatisant les différentes expériences
├── requirements.txt    # Dépendances (tensorflow, mlflow, numpy, matplotlib)
├── report_tp2.pdf      # Rapport d'analyse théorique et pratique
└── README.md           # Documentation du projet
```

## Prérequis

*   Python 3.8 ou plus.
*   Accès à une interface graphique (pour visualiser les courbes MLflow sur `localhost`).
*   Installation de MLflow pour le tracking.

## Installation

1.  **Récupérer le projet :**
    ```bash
    git clone https://github.com/ud-2/TP_DL.git
    cd TP_DL
    git checkout tp2
    ```

2.  **Préparer l'environnement virtuel :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Lancer la suite d'expériences
Le script `run_experiments.py` exécute successivement plusieurs entraînements (modèle de base, modèle régularisé, comparaison des optimiseurs, et batch normalization) :
```bash
python run_experiments.py
```

### 2. Analyser les résultats dans MLflow
Pour comparer les courbes d'apprentissage et les métriques finales de chaque configuration :
```bash
mlflow ui
```
Ensuite, ouvrez votre navigateur à l'adresse [http://localhost:5000](http://localhost:5000).

## Concepts Clés abordés

*   **Biais vs Variance** : Compréhension de l'écart entre l'erreur d'entraînement et l'erreur de validation.
*   **Régularisation L2** : Pénalisation des poids de grande magnitude pour simplifier le modèle.
*   **Dropout** : Désactivation aléatoire de neurones pour forcer la redondance des caractéristiques apprises.
*   **Batch Normalization** : Normalisation des entrées de chaque couche pour lutter contre le "Internal Covariate Shift".
*   **Optimiseurs Adaptatifs** : Pourquoi Adam est souvent le choix par défaut grâce à la combinaison du momentum et du RMSprop.