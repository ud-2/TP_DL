import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt

# Synthetic dataset
def generate_tp_data(n_samples=1500, seq_len=100):
    t = np.linspace(0, 200, n_samples + seq_len)
    series = np.sin(t) * np.cos(t/3) + 0.02 * t + np.random.normal(0, 0.05, t.shape)
    X, y = [], []
    for i in range(n_samples):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y)

# Custom attention layer
class SimpleAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W shape: (hidden_dim, 1)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        # b shape: (seq_len, 1)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        
        super(SimpleAttention, self).build(input_shape)

    def call(self, x):
        # Score calculation
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        # Softmax for alignment weights
        alignment_weights = tf.nn.softmax(e, axis=1)
        # Context vector: weighted sum
        context_vector = tf.reduce_sum(x * alignment_weights, axis=1)
        
        return context_vector, alignment_weights

# Research Challenge (Temporal Latent Space - H-TAP)
class TemporalTransformerBlock(layers.Layer):
    """Manual Implementation for the Transformer logic"""
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = layers.Dense(embed_dim, activation="relu")

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm(inputs + attn_output)
        return self.ffn(out1)

# Comparison models (Baseline TAP vs Improved H-TAP)
def build_baseline_tap(seq_len):
    inputs = layers.Input(shape=(seq_len, 1))
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(inputs)
    
    # Hybrid RNN + Attention
    context, _ = SimpleAttention()(x)
    outputs = layers.Dense(1)(context)
    
    return keras.Model(inputs, outputs, name="Baseline_RNN_Attention")

def build_improved_htap(seq_len):
    inputs = layers.Input(shape=(seq_len, 1))
    x = layers.Dense(64)(inputs)
    
    # Amelioration : Temporel Bloc
    x = TemporalTransformerBlock(embed_dim=64, num_heads=4)(x)
    context, weights = SimpleAttention()(x)
    outputs = layers.Dense(1)(context)
    
    return keras.Model(inputs, outputs, name="Improved_H_TAP"), weights

# Main: Loop, MLOps Tracking & Visualization
if __name__ == "__main__":
    SEQ_LEN = 100
    X, y = generate_tp_data(seq_len=SEQ_LEN)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    mlflow.set_experiment("TP5_Sequence_Modeling")

    for mode in ["Baseline", "Improved"]:
        with mlflow.start_run(run_name=f"Run_{mode}"):
            if mode == "Baseline":
                model = build_baseline_tap(SEQ_LEN)
            else:
                model, _ = build_improved_htap(SEQ_LEN)
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=32, verbose=0)
            
            # Evaluation
            mse, mae = model.evaluate(X_test, y_test, verbose=0)
            
            # MLOps : Attention calculation
            mlflow.log_param("architecture", mode)
            mlflow.log_metric("MSE", mse)
            print(f"Model {mode} -> MSE: {mse:.4f}")

    # Visualization
    htap_model, att_layer = build_improved_htap(SEQ_LEN)
    
    # Attention weights extraction for sample
    weight_model = keras.Model(inputs=htap_model.input, outputs=att_layer)
    sample_weights = weight_model.predict(X_test[0:1])
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(SEQ_LEN), sample_weights[0].flatten(), color='green', alpha=0.7)
    plt.title("Visualization of Attention Weights (Scaled Dot-Product)")
    plt.xlabel("Time Steps (Entry sequence)")
    plt.ylabel("Weights (Importance)")
    plt.savefig("attention_viz.png")
    print("Visualization save in 'attention_viz.png'.")