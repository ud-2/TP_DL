# TP5 : Modélisation de Séquences et Mécanismes d'Attention

Ce projet constitue le cinquième Travail Pratique du module Deep Learning. Il explore l'évolution du traitement des séquences, des RNNs classiques aux mécanismes d'attention sophistiqués, et se conclut par un défi de recherche sur les modèles d'espace latent temporel.

## Objectifs du Projet

1.  **Implémentation "from scratch"** d'une couche de *Scaled Dot-Product Attention*.
2.  **Hybridation RNN-Attention** pour la prédiction de séries temporelles complexes.
3.  **Défi de Recherche** : Amélioration de l'architecture TAP (*Temporal Latent Space Modeling*) pour garantir la cohérence à long terme via le modèle **H-TAP**.
4.  **Rédaction Scientifique** : Production d'un article de 4 pages au format NeurIPS/ICLR.

## Structure du Répertoire

```text
.
├── sequence_attention.py       # Script principal (Entraînement, Comparaison, Visualisation)
├── main.pdf                    # Article scientifique (4 pages) détaillant la recherche
├── attention_viz.png           # Visualisation des poids d'attention (Partie 1 & 2)
├── research_attention_plot.png  # Analyse de l'attention du modèle H-TAP (Partie 3)
├── requirements.txt            # Dépendances du projet
└── mlruns/                     # Dossier local de suivi MLflow (généré à l'exécution)
```

## Installation et Utilisation

### Prérequis
*   Python 3.8+
*   TensorFlow 2.x
*   MLflow
*   Matplotlib & NumPy

### 1. Installation des dépendances
```bash
pip install tensorflow mlflow matplotlib numpy
```

### 2. Exécution des expériences
Le script `sequence_attention.py` exécute automatiquement la comparaison entre le modèle de base (Baseline TAP) et notre modèle amélioré (H-TAP) :
```bash
python sequence_attention.py
```

### 3. Suivi MLOps avec MLflow
Pour visualiser les performances comparatives (MSE, MAE) et la métrique personnalisée **Attention Span**, lancez l'interface MLflow :
```bash
mlflow ui
```
Puis ouvrez `http://localhost:5000` dans votre navigateur.

## Résumé de la Recherche (Modèle H-TAP)

Le cœur de ce TP réside dans l'amélioration du modèle **TAP (ArXiv:2102.05095)**. 

### Le Problème
Les modèles d'espace latent classiques utilisent des transitions récurrentes (GRU/LSTM) qui souffrent de dérive temporelle. Sur de longues séquences (ex: Moving MNIST à 100 trames), les objets perdent leur cohérence structurelle.

### Notre Solution : H-TAP
Nous avons implémenté un bloc **Temporal Transformer** personnalisé qui remplace la transition récursive par une consultation globale de l'historique latent. 
*   **Mécanisme** : Scaled Dot-Product Attention.
*   **Avantage** : Accès direct aux trames clés du passé ($O(1)$ distance de gradient).
*   **Résultat** : Une réduction de la MSE de près de 25% sur les horizons lointains et une stabilité visuelle accrue.

## Visualisation
Le projet génère des graphiques montrant la distribution des poids d'attention. Une attention "étalée" sur toute la séquence (Attention Span élevé) confirme que le modèle utilise effectivement les informations lointaines pour stabiliser ses prédictions actuelles.