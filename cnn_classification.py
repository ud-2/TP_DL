import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Préparation des données CIFAR-10
def load_data():
    print("Chargement de CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    NUM_CLASSES = 10
    INPUT_SHAPE = x_train.shape[1:] # (32, 32, 3)

    # Normalisation
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-Hot Encoding
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    return (x_train, y_train), (x_test, y_test), INPUT_SHAPE, NUM_CLASSES

# 2. Exercice 1 : Architecture CNN Classique
def build_basic_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Classification
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. Exercice 2 : Blocs Résiduels (ResNet)
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    shortcut = x
    
    # Chemin principal
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)
    
    # Si la dimension change (stride > 1), on adapte le raccourci avec une conv 1x1
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
    
    z = layers.Add()([shortcut, y])
    z = layers.Activation('relu')(z)
    return z

def build_resnet_mini(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # 3 Blocs résiduels consécutifs
    x = residual_block(x, 32)
    x = residual_block(x, 64, stride=2) # Downsampling
    x = residual_block(x, 64)
    
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# 4. Exercice 4 : Neural Style Transfer (Extracteur)
def create_style_extractor():
    # Chargement de VGG16 pré-entraîné
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    return keras.Model(inputs=vgg.input, outputs=outputs)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test), shape, classes = load_data()

    # Entraînement du CNN Classique
    model = build_basic_cnn(shape, classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("\n--- Entraînement du CNN Classique ---")
    model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)
    
    loss, acc = model.evaluate(x_test, y_test)
    print(f"\nPrécision finale sur CIFAR-10 : {acc:.4f}")