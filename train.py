import os
import tensorflow as tf
import json
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    logger.info(f"GPU device(s) available: {physical_devices}")
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.warning(f"Could not set memory growth: {e}")
else:
    logger.info("No GPU available, using CPU.")

from tensorflow.keras import layers
from keras.src.utils import plot_model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

from preprocess import preprocess_data

MODEL_PATH = 'model/cnn_pruning.keras'
CLASS_NAME_PATH = 'model/class_names.npy'
HISTORY_PATH = 'model/training_history.json'
DATA_PATH = 'data/archive/indoorCVPR_09/Images'
MODEL_ARCH_PATH = 'model/model_architecture.png'

for path in [MODEL_PATH, CLASS_NAME_PATH, HISTORY_PATH, MODEL_ARCH_PATH, 'model/best_model.keras']:
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise

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
            progress = (self.current_step - self.begin_step) / (self.end_step - self.begin_step)
            sparsity = self.initial_sparsity + (self.final_sparsity - self.initial_sparsity) * progress

            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    weights = layer.get_weights()
                    if len(weights) > 0:
                        kernel = weights[0]
                        flat_weights = tf.reshape(kernel, [-1])
                        num_weights = tf.size(flat_weights).numpy()
                        num_to_prune = int(sparsity * num_weights)
                        abs_weights = tf.abs(flat_weights)
                        _, indices = tf.nn.top_k(-abs_weights, k=num_to_prune)
                        mask = tf.ones_like(flat_weights, dtype=tf.float32)
                        mask = tf.tensor_scatter_nd_update(mask, tf.expand_dims(indices, 1), tf.zeros(num_to_prune))
                        mask = tf.reshape(mask, kernel.shape)
                        new_kernel = kernel * mask
                        if len(weights) > 1:
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

def train_model(train_dataset, val_dataset, train_steps, val_steps, class_names):
    num_classes = len(class_names)
    device = '/GPU:0' if physical_devices else '/CPU:0'
    logger.info(f"Using device: {device}")

    with tf.device(device):
        model = create_model(num_classes)

        try:
            plot_model(model, to_file=MODEL_ARCH_PATH, show_shapes=True, show_layer_names=True)
            logger.info(f'Model architecture saved to {MODEL_ARCH_PATH}')
        except Exception as e:
            logger.warning(f"Failed to save model architecture: {e}")

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model Summary:")
        model.summary(print_fn=lambda x: logger.info(x))

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
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=50,
            callbacks=[checkpoint, early_stopping, reduce_lr, pruning_callback]
        )

        try:
            model.save(MODEL_PATH)
            logger.info(f'Model saved to {MODEL_PATH}')
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise

        history_dict = history.history
        try:
            with open(HISTORY_PATH, 'w') as f:
                json.dump(history_dict, f)
            logger.info(f'Training history saved to {HISTORY_PATH}')
        except Exception as e:
            logger.error(f"Failed to save training history: {e}")
            raise

    return model

if __name__ == "__main__":
    try:
        (train_dataset, test_dataset, val_dataset,
         train_steps, test_steps, val_steps,
         class_names) = preprocess_data(DATA_PATH, seed=42)

        np.save(CLASS_NAME_PATH, class_names)
        model = train_model(train_dataset, val_dataset, train_steps, val_steps, class_names)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise