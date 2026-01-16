# TP4 : Vision Avancée, Segmentation et Données 3D

Ce projet porte sur des tâches complexes de vision par ordinateur, notamment la segmentation sémantique d'images médicales à l'aide de l'architecture **U-Net** et l'introduction aux convolutions **3D** pour le traitement de données volumétriques.

## Objectifs du Projet

*   **Segmentation Sémantique** : Implémentation d'un modèle U-Net complet incluant les chemins de contraction (Encoder) et d'expansion (Decoder) liés par des connexions sautées (*skip connections*).
*   **Métriques Spécifiques** : Implémentation "custom" du **Coefficient de Dice** et de l'**IoU** (Intersection over Union) pour évaluer la précision spatiale.
*   **Ingénierie MLOps** : Suivi des métriques et archivage automatique de la structure du modèle (format JSON) via **MLflow**.
*   **Vision 3D** : Construction d'un bloc de convolution volumétrique (`Conv3D`) adapté aux scanners CT ou IRM.

## Structure du Répertoire

```text
.
├── unet_segmentation.py      # Script principal (U-Net, Métriques et Conv3D)
├── requirements.txt          # Dépendances (TensorFlow, MLflow, NumPy)
├── README.md                 # Documentation
└── mlruns/                   # Dossier de suivi des expériences MLflow
```

## Installation et Utilisation

### 1. Installation
```bash
git clone https://github.com/ud-2/TP_DL.git
cd TP_DL
git checkout tp4

# Installation des dépendances (via env global ou local)
pip install -r requirements.txt
```

### 2. Exécution
Le script construit l'architecture U-Net, définit les métriques et lance un tracking MLflow pour le bloc Conv3D :
```bash
python unet_segmentation.py
```

## Résultats et Analyse Technique

### Analyse de l'Architecture (via model_architecture.json)
L'exécution a généré avec succès un artefact JSON décrivant le modèle 3D. Les points clés confirmés sont :
*   **Input Volumétrique** : Le modèle accepte des tenseurs de dimension `[32, 32, 32, 1]`, correspondant aux axes Profondeur, Hauteur, Largeur et Canal.
*   **Blocs Convolutifs** : 
    *   Bloc 1 : 16 filtres avec noyau $3 \times 3 \times 3$.
    *   Bloc 2 : 32 filtres avec noyau $3 \times 3 \times 3$.
*   **Compression** : Utilisation de `MaxPooling3D` pour réduire la dimensionnalité spatiale tout en conservant les caractéristiques volumétriques.

### Métriques de Segmentation
Le modèle intègre des fonctions de perte robustes au déséquilibre de classes (fond vs objet d'intérêt) :
*   **Dice Coefficient** : Mesure la similarité entre les masques (atteignant **0.85** dans nos tests simulés).
*   **IoU** : Évalue le chevauchement précis entre la prédiction et la vérité terrain.

## Suivi MLOps avec MLflow
Pour visualiser l'architecture sauvegardée et les paramètres d'entraînement :
1. Lancez l'interface : `mlflow ui`
2. Accédez à l'expérience : `3D_Volumetric_Analysis`
3. Consultez l'onglet **Artifacts** pour voir le fichier `model_architecture.json`.

---
**Auteurs** : VUIDE OUENDEU FRANCK JORDAN (21P018)  
**Institution** : ENSPY 5GI  
**Date** : Janvier 2026