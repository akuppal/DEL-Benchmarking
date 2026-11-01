import tensorflow as tf

SEED = 42
LEARNING_RATE = 0.001
ALPHA = 0.0005  # L2 regularization term
HIDDEN_LAYER_SIZES = (100, 150, 150, 100)
ACTIVATION = "relu"
N_FEATURES = 2048
FIT_KWARGS = dict(epochs=20, batch_size=256, validation_split=0.1, verbose=2)


def build_model():
    tf.keras.utils.set_random_seed(SEED)
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(N_FEATURES,))]
        + [tf.keras.layers.Dense(n, activation=ACTIVATION,
               kernel_regularizer=tf.keras.regularizers.l2(ALPHA))
           for n in HIDDEN_LAYER_SIZES]
        + [tf.keras.layers.Dense(1, activation="sigmoid",
               kernel_regularizer=tf.keras.regularizers.l2(ALPHA))]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="binary_crossentropy")
    return model


if __name__ == "__main__":
    import h5py, numpy as np
    from sklearn.model_selection import train_test_split

    with h5py.File("train_feature.h5") as hf:
        X = hf["feature"][:]
    y = np.load("labels.npy")  # same order as the features

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = build_model()
    model.fit(Xtr, ytr, **FIT_KWARGS)
    model.save("my_mlp.keras")
