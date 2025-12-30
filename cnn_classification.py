import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. PRÉPARATION DES DONNÉES CIFAR-10 (Exercice 1.2)
# ==============================================================================

print("--- Chargement et préparation des données ---")

# Chargement du jeu de données CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:]  # Devrait être (32, 32, 3)

# Normalisation des pixels (0 à 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Conversion des labels en One-Hot Encoding
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

print(f"Shape des données d'entrée : {INPUT_SHAPE}")
print(f"Shape des labels (One-Hot) : {y_train.shape}")


# ==============================================================================
# 2. ARCHITECTURE CNN CLASSIQUE (Exercice 2.1)
# ==============================================================================

def build_basic_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Bloc Convolutif 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloc Convolutif 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Transition vers les couches denses
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        # Couche de sortie
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

print("\n--- Entraînement du CNN Classique ---")
model_basic = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
model_basic.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

# Entraînement (10 époques comme demandé)
history_basic = model_basic.fit(
    x_train, y_train, 
    batch_size=64, 
    epochs=10, 
    validation_split=0.1
)

# Évaluation sur l'ensemble de test
test_loss, test_acc = model_basic.evaluate(x_test, y_test)
print(f"\nPrécision sur les données de test (CNN Classique) : {test_acc:.4f}")


# ==============================================================================
# 3. INTRODUCTION AUX BLOCS RÉSIDUELS (ResNets - Exercice 2.2)
# ==============================================================================

def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """
    Implémentation d'un bloc résiduel simplifié.
    """
    shortcut = x
    
    # Chemin principal (Main path)
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    
    # Chemin de saut (Skip Connection)
    # Si les dimensions ne correspondent pas (à cause du stride ou du nombre de filtres),
    # on applique une convolution 1x1 pour ajuster le raccourci.
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
    
    # Addition du raccourci et du chemin principal
    z = layers.Add()([shortcut, y])
    z = layers.Activation('relu')(z)
    return z

def build_resnet_small(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Couche initiale
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Ajout de 3 blocs résiduels consécutifs
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2) # On augmente les filtres et réduit la taille spatiale
    x = residual_block(x, 64)
    
    # Global Average Pooling au lieu de Flatten (pratique courante en ResNet)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

print("\n--- Construction du modèle ResNet ---")
model_resnet = build_resnet_small(INPUT_SHAPE, NUM_CLASSES)
model_resnet.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
model_resnet.summary()

# Note : L'entraînement du ResNet est lancé ici de la même manière si besoin.
# model_resnet.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)


# ==============================================================================
# 4. NEURAL STYLE TRANSFER (Exercice 4 - Structure de base)
# ==============================================================================

def setup_style_transfer():
    """
    Prépare les éléments pour le transfert de style (VGG16).
    Note : Cette partie nécessite des images externes pour fonctionner.
    """
    print("\n--- Configuration du Style Transfer (VGG16) ---")
    
    # Chargement de VGG16 pré-entraîné sans le haut (couches denses)
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Définition des couches de contenu et de style
    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1', 
        'block2_conv1', 
        'block3_conv1', 
        'block4_conv1', 
        'block5_conv1'
    ]
    
    # Création du modèle extracteur
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    extractor = keras.Model(inputs=vgg.input, outputs=outputs)
    
    return extractor

# Initialisation de l'extracteur pour le rapport
# extractor = setup_style_transfer()

if __name__ == "__main__":
    print("\nTP3 terminé avec succès.")