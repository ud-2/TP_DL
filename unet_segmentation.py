import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
import mlflow
import mlflow.tensorflow
import numpy as np

# 1. Métriques Personnalisées
def dice_coeff(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# 2. Architecture U-Net
def conv_block(input_tensor, num_filters):
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_unet(input_shape=(128, 128, 1)):
    inputs = layers.Input(input_shape)

    # ENCODER (Contracting Path)
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # BRIDGE / BOTTLENECK
    b = conv_block(p3, 256)

    # DECODER (Expansive Path)
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    u1 = layers.concatenate([u1, c3])
    d1 = conv_block(u1, 128)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = layers.concatenate([u2, c2])
    d2 = conv_block(u2, 64)

    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = layers.concatenate([u3, c1])
    d3 = conv_block(u3, 32)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)
    
    return keras.Model(inputs, outputs)

# 3. Exercice 3 : Conv3D pour Données Volumétriques
def simple_conv3d_block(input_shape=(32, 32, 32, 1)):
    inputs = layers.Input(input_shape)
    
    # Premier bloc 3D
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    # Deuxième bloc 3D
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs)

# 4. Pipeline d'ingénierie (MLOps)
if __name__ == "__main__":
    # Analyse U-Net
    unet = build_unet()
    unet.compile(optimizer='adam', 
                 loss='binary_crossentropy', 
                 metrics=[dice_coeff, iou_metric])
    
    # Analyse Conv3D avec MLflow
    mlflow.set_experiment("3D_Volumetric_Analysis")
    
    with mlflow.start_run(run_name="Conv3D_Baseline"):
        model_3d = simple_conv3d_block()
        
        # Logging de l'architecture
        mlflow.log_dict({"config": model_3d.to_json()}, "artifacts/model_architecture.json")
        
        # Logging des hyperparamètres
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("filters_start", 16)
        
        # Simulation de métriques
        mlflow.log_metric("final_val_dice", 0.85)
        
        print("U-Net construit et suivi MLflow pour Conv3D terminé.")