import os

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"GPU device(s) available: {physical_devices}")
    try:
        # Enable memory growth for each GPU
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Warning: Could not set memory growth: {e}")
else:
    print("No GPU available, using CPU.")

from tensorflow.keras import layers
from keras.src.utils import plot_model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

MODEL_PATH = 'model/cnn_pruning.keras'
CLASS_NAME_PATH = 'model/class_names.npy'
DATA_PATH = 'archive/indoorCVPR_09/Images'
MODEL_ARCH_PATH = 'model/model_architecture.png'

for path in [MODEL_PATH, CLASS_NAME_PATH, MODEL_ARCH_PATH]:
    directory = os.path.dirname(path)  # Extract directory part of the path
    if directory and not os.path.exists(directory):  # Check if directory is non-empty and doesn't exist
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


# Custom pruning callback
class PruningCallback(Callback):
    def __init__(self, initial_sparsity=0.2, final_sparsity=0.8, begin_step=2000, end_step=10000):
        super(PruningCallback, self).__init__()
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.current_step = 0

    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        if self.current_step >= self.begin_step and self.current_step <= self.end_step:
            # Calculate current sparsity using PolynomialDecay-like schedule
            progress = (self.current_step - self.begin_step) / (self.end_step - self.begin_step)
            sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress

            # Apply pruning to convolutional layers
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    weights = layer.get_weights()
                    if len(weights) > 0:  # Check if layer has weights
                        kernel = weights[0]  # Shape: [height, width, in_channels, out_channels]
                        # Flatten weights for pruning
                        flat_weights = tf.reshape(kernel, [-1])
                        num_weights = tf.size(flat_weights)
                        num_to_prune = int(sparsity * num_weights)
                        # Get indices of smallest absolute weights
                        abs_weights = tf.abs(flat_weights)
                        _, indices = tf.nn.top_k(-abs_weights, k=num_to_prune)  # Negative for smallest
                        mask = tf.ones_like(flat_weights, dtype=tf.float32)
                        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), tf.zeros(num_to_prune))
                        # Reshape mask back to kernel shape
                        mask = tf.reshape(mask, kernel.shape)
                        # Apply mask to weights (preserve biases if present)
                        new_kernel = kernel * mask
                        if len(weights) > 1:  # Include bias
                            layer.set_weights([new_kernel, weights[1]])
                        else:
                            layer.set_weights([new_kernel])


def create_model(num_classes, input_shape=(224, 224, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='last_conv')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model


def train_model(train_generator, val_generator, class_names):
    num_classes = len(class_names)

    device = '/GPU:0' if physical_devices else '/CPU:0'
    print(device)
    with tf.device(device):
        model = create_model(num_classes)

    plot_model(
        model,
        to_file=MODEL_ARCH_PATH,
        show_shapes=True,
        show_layer_names=True
    )
    print(f'Model architecture saved to {MODEL_ARCH_PATH}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Print model summary before training
    print("Model Summary:")
    model.summary()

    checkpoint = ModelCheckpoint(
        'model/best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    pruning_callback = PruningCallback(
        initial_sparsity=0.2,
        final_sparsity=0.8,
        begin_step=2000,
        end_step=10000
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[checkpoint, early_stopping, reduce_lr, pruning_callback]
    )

    model.save(MODEL_PATH)
    return model


if __name__ == "__main__":
    from preprocess import preprocess_data  # Ensure this is your TensorFlow preprocess script

    train_gen, val_gen, class_names = preprocess_data(DATA_PATH, seed=42)
    np.save(CLASS_NAME_PATH, class_names)

    model = train_model(train_gen, val_gen, class_names)