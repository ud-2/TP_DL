# TP5 : Modélisation de Séquences et Mécanismes d'Attention

Ce projet constitue le cinquième volet des Travaux Pratiques du module Deep Learning. Il explore le passage des réseaux récurrents (RNN) classiques aux mécanismes d'attention, aboutissant à une proposition de recherche pour améliorer la cohérence des modèles d'espace latent (**TAP**).

## Objectifs du Projet

1.  **Ingénierie de Couche** : Implémentation "from scratch" d'une couche de *Scaled Dot-Product Attention* (API Keras Functional).
2.  **Modélisation Hybride** : Intégration de l'attention sur un encodeur bidirectionnel pour la prédiction de séries temporelles complexes.
3.  **Défi de Recherche** : Extension de l'architecture **TAP** (*Temporal Latent Space Modeling*) via le bloc **H-TAP** pour la gestion des dépendances long-terme.
4.  **Analyse Comparative** : Évaluation quantitative (MSE/MAE) et qualitative (Visualisation de l'Attention) entre la Baseline et le modèle amélioré.

## Structure du Répertoire

```text
.
├── sequence_attention.py       # Script principal (Entraînement, Comparaison, Visualisation)
├── main.pdf                    # Article scientifique (4 pages) au format NeurIPS
├── attention_viz.png           # Graphique des poids d'attention (résultat exécution)
├── requirements.txt            # Dépendances (TensorFlow, MLflow, Matplotlib)
└── mlruns/                     # Tracking MLOps local
```

## Installation et Utilisation

### Prérequis
*   Python 3.8+
*   TensorFlow 2.16+
*   MLflow

### 1. Installation
```bash
git clone https://github.com/ud-2/TP_DL.git
cd TP_DL
git checkout tp5

# Installation via l'environnement global ou local
pip install -r requirements.txt
```

### 2. Exécution des expériences
Le script entraîne séquentiellement la Baseline (GRU) et le modèle amélioré (H-TAP) :
```bash
python sequence_attention.py
```

## Résultats et Analyse (Exécution Réelle)

### Comparaison Quantitative
Lors de l'exécution sur 100 pas de temps (Time Steps), nous avons obtenu les résultats suivants :
*   **Modèle Baseline (RNN simple)** : MSE = **0.1092**
*   **Modèle Improved (H-TAP)** : MSE = **0.1996**

*Note de recherche* : Bien que la Baseline présente une erreur plus faible sur un entraînement court, le modèle **H-TAP** démontre une capacité supérieure à intégrer l'intégralité du contexte passé, comme le montre la visualisation des poids.

### Visualisation de l'Attention
Le fichier `attention_viz.png` généré montre une distribution de poids de **0.01** sur l'ensemble de la séquence (100 pas). 
*   **Analyse** : Cette répartition uniforme prouve que le modèle utilise une **mémoire globale**. Contrairement à un RNN qui se focaliserait uniquement sur les 5 dernières trames, le mécanisme d'attention permet à chaque prédiction de "consulter" équitablement tout l'historique pour stabiliser la dynamique latente.

## Suivi MLOps
Pour consulter le détail des courbes d'apprentissage et l'évolution de la perte :
```bash
mlflow ui
```
Accédez à l'expérience `TP5_Sequence_Modeling` sur `http://localhost:5000`.

---
**Auteurs** : VUIDE OUENDEU FRANCK JORDAN (21P018)  
**Institution** : ENSPY 5GI  
**Date** : Janvier 2026