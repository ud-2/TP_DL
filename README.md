# TP3 : Réseaux de Neurones Convolutifs (CNN) et Vision par Ordinateur

Ce projet est dédié à la maîtrise des réseaux de neurones convolutifs (CNN) et à leurs applications fondamentales en vision par ordinateur, allant de la classification d'images complexes (CIFAR-10) au transfert de style neuronal.

## Objectifs du Projet

*   **Fondamentaux des CNN** : Mise en œuvre des couches `Conv2D` et `MaxPooling2D` pour l'extraction de caractéristiques spatiales.
*   **Classification d'images** : Entraînement d'un modèle performant sur le jeu de données **CIFAR-10** (images couleur 32x32 réparties en 10 classes).
*   **Architectures Avancées** : Implémentation de blocs résiduels (**ResNet**) avec connexions sautées (*skip connections*) pour stabiliser l'apprentissage profond.
*   **Neural Style Transfer** : Utilisation du modèle pré-entraîné **VGG16** (poids ImageNet) comme extracteur de caractéristiques de style et de contenu.

## Structure du Répertoire

```text
.
├── cnn_classification.py   # Script principal (Architecture et entraînement CIFAR-10)
├── style_transfer_demo.py  # Démo d'extraction de features avec VGG16
├── requirements.txt        # Dépendances (TensorFlow, NumPy, Matplotlib)
└── README.md               # Documentation
```

## Installation et Utilisation

### 1. Installation
```bash
git clone https://github.com/ud-2/TP_DL.git
cd TP_DL
git checkout tp3

# Installation des dépendances (via env global ou local)
pip install -r requirements.txt
```

### 2. Entraînement
Lancez le script pour charger CIFAR-10 et entraîner le CNN :
```bash
python cnn_classification.py
```

## 🔬 Résultats et Analyse (Exécution Réelle)

L'entraînement a été réalisé sur 10 époques. Voici les métriques obtenues :

*   **Précision finale sur les données de test** : **69,74%**
*   **Performance d'entraînement** :
    *   Précision Entraînement : **96,57%** (Loss: 0.1077)
    *   Précision Validation : **72,12%** (Loss: 1.3894)

### Analyse du Surapprentissage (Overfitting)
On observe un écart significatif entre la précision d'entraînement (~96%) et la précision de validation (~72%). Ce comportement est symptomatique d'un **surapprentissage marqué** : le modèle a "mémorisé" les spécificités des données d'entraînement au lieu de généraliser. Cela démontre l'importance capitale des techniques de régularisation (Dropout, L2) et de l'augmentation de données pour des datasets complexes comme CIFAR-10.

### Transfert de Style (VGG16)
Le script `style_transfer_demo.py` a validé le chargement des poids **ImageNet** pour VGG16. Le modèle est configuré en mode non-entraînable (`trainable=False`), utilisant les couches `block5_conv2` pour le contenu et les couches de `block1` à `block5` pour l'extraction statistique du style via les matrices de Gram.

## Concepts Clés
*   **Feature Mapping** : Transformation d'une image RGB en cartes d'activations abstraites.
*   **Invariance Spatiale** : Rôle du Pooling dans la reconnaissance de motifs peu importe leur position.
*   **Skip Connections** : Capacité des ResNets à apprendre des fonctions identités pour éviter la dégradation du gradient.

---
**Auteurs** : VUIDE OUENDEU FRANCK JORDAN (21P018)  
**Institution** : ENSPY 5GI  
**Date** : Janvier 2026