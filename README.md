# Projet : Cycle de Vie et Ingénierie des Modèles de Deep Learning

Ce dépôt GitHub contient l'intégralité des travaux pratiques réalisés dans le cadre du cursus d'ingénierie en Deep Learning. Il documente le cycle de vie complet d'un modèle : depuis la conception mathématique initiale jusqu'au déploiement industriel, en passant par l'optimisation avancée, la vision par ordinateur et la modélisation de séquences.

Le projet est structuré de manière modulaire, chaque étape résidant sur sa propre branche Git pour une isolation parfaite des environnements et des rapports.

## Structure du Dépôt

Pour consulter le travail spécifique à chaque étape, utilisez les branches suivantes :

*   **`main`** : Présentation générale et architecture du projet.
*   **`tp1`** : Conception, MLOps de base et déploiement (MNIST, Flask, Docker).
*   **`tp2`** : Ingénierie de la performance (Régularisation, Optimiseurs, Batch Norm).
*   **`tp3`** : Vision par Ordinateur classique et résiduelle (CIFAR-10, ResNet, VGG16).
*   **`tp4`** : Vision Avancée et Imagerie Médicale (U-Net, Métriques Spatiales, Conv3D).
*   **`tp5`** : Modélisation de Séquences et Recherche (Attention, H-TAP, Transformers).

---

## Contenu des Travaux Pratiques

### 1. [Branche tp1] : De la Conception au Déploiement
Mise en place d'un pipeline complet de production :
*   **Modélisation** : Réseau dense sur MNIST avec une précision de **97.83%**.
*   **Serving** : Création d'une API REST avec **Flask**.
*   **Industrialisation** : Conteneurisation de l'application via **Docker**.

### 2. [Branche tp2] : Amélioration et Robustesse
Techniques avancées pour stabiliser l'apprentissage et éviter le surapprentissage :
*   **Optimisation** : Comparaison comparative d'**Adam**, **RMSprop** et **SGD**.
*   **Régularisation** : Implémentation du Dropout et de la régularisation L2.
*   **Normalisation** : Utilisation de la **Batch Normalization** pour accélérer la convergence.

### 3. [Branche tp3] : CNN et Architectures Résiduelles
Transition vers le traitement d'images couleur et les réseaux profonds :
*   **CNN** : Classification sur CIFAR-10 (Précision : **69.74%**).
*   **ResNet** : Implémentation manuelle de blocs résiduels (*skip connections*).
*   **Feature Extraction** : Utilisation de **VGG16** pour le transfert de style neuronal.

### 4. [Branche tp4] : Vision Avancée et Données 3D
Tâches complexes de segmentation et manipulation de volumes :
*   **U-Net** : Architecture Encodeur-Décodeur pour la segmentation sémantique.
*   **Métriques** : Implémentation du coefficient de **Dice** et de l'**IoU**.
*   **Données 3D** : Utilisation de `Conv3D` pour le traitement de données volumétriques.

### 5. [Branche tp5] : Séquences et Mécanismes d'Attention
Modélisation temporelle et projet de recherche final :
*   **Attention** : Implémentation "from scratch" de la *Scaled Dot-Product Attention*.
*   **Recherche H-TAP** : Amélioration du modèle TAP (Temporal Latent Space) pour la cohérence vidéo à long terme.
*   **Visualisation** : Analyse qualitative des poids d'attention.

---

## Comment Accéder au Code

Après avoir cloné le dépôt, vous pouvez naviguer entre les TPs :

```bash
# Cloner le dépôt
git clone https://github.com/ud-2/TP_DL.git
cd TP_DL

# Accéder au TP souhaité (ex: TP4)
git checkout tp4

# Pour revenir à l'accueil
git checkout main
```

## 🛠 Technologies Utilisées

*   **Frameworks** : TensorFlow, Keras, Flask.
*   **MLOps** : MLflow (Tracking & Artifacts).
*   **Ops** : Docker, Git.
*   **Analyse** : NumPy, Matplotlib, OpenCV, PIL.
*   **Rédaction** : LaTeX (Overleaf).

---
**Réalisation** : VUIDE OUENDEU FRANCK JORDAN (21P018)  
**Institution** : École Nationale Supérieure Polytechnique de Yaoundé (**ENSPY**)  
**Promotion** : 5ème Année Génie Informatique (5GI)