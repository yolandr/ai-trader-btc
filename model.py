"""
model.py
Arsitektur Neural Network multi-layer (LSTM + Dense) untuk
klasifikasi tren BTC: DOWN / SIDEWAYS / UP.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers

import config


def build_model(input_shape: tuple) -> tf.keras.Model:
    """
    input_shape = (window, n_features)

    Arsitektur:
      Input -> LSTM(128, return_sequences) -> Dropout
            -> LSTM(64, return_sequences)  -> Dropout
            -> LSTM(32)                     -> Dropout
            -> Dense(32, relu)               -> Dropout
            -> Dense(16, relu)
            -> Dense(3, softmax)   # DOWN / SIDEWAYS / UP
    """
    inputs = layers.Input(shape=input_shape, name="price_sequence")

    x = inputs
    lstm_units = config.LSTM_UNITS
    for i, units in enumerate(lstm_units):
        return_seq = i < len(lstm_units) - 1  # layer terakhir tidak return sequence
        x = layers.LSTM(
            units,
            return_sequences=return_seq,
            kernel_regularizer=regularizers.l2(config.L2_REG),
            name=f"lstm_{i+1}"
        )(x)
        x = layers.Dropout(config.DROPOUT_RATE, name=f"dropout_lstm_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

    for i, units in enumerate(config.DENSE_UNITS):
        x = layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        x = layers.Dropout(config.DROPOUT_RATE / 2, name=f"dropout_dense_{i+1}")(x)

    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="trend_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="btc_trend_predictor")

    optimizer = optimizers.Adam(learning_rate=config.LEARNING_RATE)  # learning rate 0.001

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    m = build_model(input_shape=(config.LOOKBACK_WINDOW, 19))
    m.summary()