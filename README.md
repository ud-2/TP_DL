# TP3 : Réseaux de Neurones Convolutifs (CNN) et Vision

Ce TP explore les architectures spécialisées dans le traitement d'images et les techniques modernes de vision par ordinateur.

## Objectifs du Projet
*   **CNN Classique** : Construction d'un modèle pour le dataset **CIFAR-10** (images couleur 32x32).
*   **ResNets** : Implémentation manuelle de **blocs résiduels** (connexions sautées/skip connections).
*   **Applications Avancées** :
    *   Concepts de **Segmentation** (U-Net) et **Détection d'objets** (Bounding Boxes).
    *   **Neural Style Transfer** : Utilisation d'un modèle pré-entraîné (**VGG16**) pour extraire le style et le contenu d'images.

## Structure du Projet
```text
.
├── cnn_classification.py   # Architecture CNN et ResNet pour CIFAR-10
├── style_transfer_demo.py  # Script d'extraction de caractéristiques (VGG16)
├── requirements.txt
└── README.md
```

## Utilisation
Entraîner le classifieur d'images :
```bash
python cnn_classification.py
```

## Concepts abordés
*   Opérations de Convolution et de Pooling.
*   Problème de la disparition du gradient dans les réseaux profonds.
*   Matrices de Gram pour la représentation du style.
*   Transfer Learning (utilisation de poids ImageNet).