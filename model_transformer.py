import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout_rate=0.15):
    # Self-attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(x, x)
    x = Dropout(dropout_rate)(x)
    res = x + inputs

    # Feed-forward
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)  # Back to relu for stability
    x = Dropout(dropout_rate)(x)
    x = Dense(inputs.shape[-1])(x)

    return x + res


def build_model(input_shape, lr=1e-3):

    inputs = Input(shape=input_shape)

    # Less aggressive architecture
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout_rate=0.15)
    x = transformer_encoder(x,      head_size=64, num_heads=4, ff_dim=128, dropout_rate=0.15)

    x = GlobalAveragePooling1D()(x)

    # Simpler classifier head
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)

    outputs = Dense(3, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model