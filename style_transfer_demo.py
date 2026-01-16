import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

def load_and_process_img(path_to_img):
    """Load, resize, and apply VGG16 preprocessing."""
    
    img = Image.open(path_to_img).resize((512, 512))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = keras.applications.vgg16.preprocess_input(img)
    
    return tf.convert_to_tensor(img)

def build_feature_extractor():
    print("Loading VGG16 model (ImageNet weights)...")
    
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Define layers for style and content extraction
    content_layers = ['block5_conv2']
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1'
    ]
    
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    return keras.Model(inputs=vgg.input, outputs=outputs)

if __name__ == "__main__":
    extractor = build_feature_extractor()
    
    print("Feature extractor built successfully.")
    print(f"Number of output layers: {len(extractor.outputs)}")

    # Extraction test using a dummy image
    dummy_img = tf.random.uniform((1, 512, 512, 3))
    features = extractor(dummy_img)

    print("\nFeature extraction completed on dummy image.")
    for i, feat in enumerate(features):
        print(f" - Output {i} shape: {feat.shape}")