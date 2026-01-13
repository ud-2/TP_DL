# TP3 : Réseaux de Neurones Convolutifs (CNN) et Vision par Ordinateur

Ce projet est dédié à la maîtrise des réseaux de neurones convolutifs (CNN) et à leurs applications fondamentales en vision par ordinateur, allant de la classification d'images complexes au transfert de style neuronal.

## Objectifs du Projet

*   **Fondamentaux des CNN** : Compréhension et implémentation des opérations de convolution et de pooling.
*   **Classification d'images** : Construction et entraînement d'un CNN sur le jeu de données **CIFAR-10** (10 classes d'images couleur 32x32).
*   **Architectures Avancées (ResNets)** : Implémentation de blocs résiduels (*skip connections*) pour permettre l'entraînement de réseaux plus profonds.
*   **Reconnaissance Visuelle** : Exploration des concepts de segmentation d'images et de détection d'objets (Bounding Boxes).
*   **Neural Style Transfer** : Utilisation d'un modèle pré-entraîné (**VGG16**) pour séparer et combiner le contenu d'une image avec le style d'une autre.

## Structure du Projet

```text
.
├── cnn_classification.py   # Script principal (Classification CIFAR-10 et ResNets)
├── style_transfer_demo.py  # Script pour l'extraction de style via VGG16
├── requirements.txt        # Dépendances (tensorflow, numpy, matplotlib, pillow)
└── README.md               # Documentation du projet
```

## Prérequis

*   Python 3.8 ou plus
*   Un environnement virtuel (recommandé)
*   Accès à Internet pour le téléchargement automatique des datasets (CIFAR-10) et des poids du modèle (VGG16).

## Installation

1.  **Cloner le dépôt et accéder au dossier :**
    ```bash
    git clone https://github.com/ud-2/TP_DL.git
    cd TP_DL
    git checkout tp3
    ```

2.  **Créer et activer un environnement virtuel :**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Sur Windows: venv\Scripts\activate
    ```

3.  **Installer les bibliothèques nécessaires :**
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

### 1. Entraînement du classifieur (CNN & ResNet)
Le script entraîne un modèle sur CIFAR-10 et évalue ses performances :
```bash
python cnn_classification.py
```

### 2. Démonstration du transfert de style
Pour charger le modèle VGG16 et préparer l'extraction des couches de style et de contenu :
```bash
python style_transfer_demo.py
```

## Concepts Clés abordés

*   **Filtres et Stride** : Rôle des noyaux de convolution dans l'extraction de caractéristiques spatiales.
*   **Architecture Résiduelle** : Comment les connexions de saut résolvent le problème de la disparition du gradient.
*   **Upsampling** : Rôle crucial dans la segmentation d'images (architecture U-Net).
*   **Matrices de Gram** : Utilisation pour représenter statistiquement le style d'une image.
*   **Feature Extraction** : Utilisation de modèles pré-entraînés sur ImageNet pour des tâches spécialisées.