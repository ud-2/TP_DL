# TP4 : Vision Avancée, Segmentation et Données 3D

Ce projet porte sur des tâches complexes de vision par ordinateur, notamment la segmentation sémantique d'images médicales à l'aide de l'architecture **U-Net** et l'introduction aux convolutions **3D** pour les données volumétriques.

## Objectifs du Projet

*   **Segmentation Sémantique** : Implémentation d'un modèle U-Net complet (Encoder, Bottleneck, Decoder) avec connexions sautées (*skip connections*).
*   **Métriques Spécifiques** : Implémentation et utilisation du coefficient de Dice et de l'IoU (Intersection over Union).
*   **MLOps** : Suivi des expérimentations et des métriques personnalisées avec **MLflow**.
*   **Données 3D** : Exploration des couches `Conv3D` pour le traitement de volumes (ex: scanners CT, IRM).

## Structure du Projet

```text
.
├── unet_segmentation.py  # Script principal (U-Net, Métriques et Conv3D)
├── requirements.txt      # Dépendances (tensorflow, mlflow, numpy)
├── README.md             # Documentation du projet
└── rapport_tp4.pdf       # Réponses théoriques et analyse des résultats
```

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/ud-2/TP_DL.git
cd TP_DL
checkout tp2

python3 -m venv venv # Si aucun environnement virtuel n'est défini
source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### 1. Entraînement et Suivi
Le script simule l'entraînement d'un U-Net et d'un bloc Conv3D. Il enregistre l'architecture et les métriques dans MLflow :
```bash
python unet_segmentation.py
```

### 2. Visualisation MLflow
Pour comparer les architectures et voir les métriques personnalisées (Dice/IoU) :
```bash
mlflow ui
```
Accédez ensuite à `http://localhost:5000`.

## Concepts Clés abordés
*   **Connexions sautées** : Concatenation des caractéristiques de l'encodeur vers le décodeur pour préserver les détails spatiaux.
*   **Déséquilibre des classes** : Utilisation du Dice Loss pour gérer les cas où l'objet à segmenter est très petit par rapport au fond.
*   **Convolutions 3D** : Extension des filtres à une troisième dimension (profondeur) pour capturer des motifs spatiaux volumétriques.