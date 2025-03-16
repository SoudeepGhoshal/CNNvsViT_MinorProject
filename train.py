import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_model_optimization as tfmot
from keras.src.utils import plot_model
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from preprocess import preprocess_data

MODEL_PATH = 'model/cnn_pruning.keras'
CLASS_NAME_PATH = 'model/class_names.npy'
DATA_PATH = 'data/indoorCVPR_09'
MODEL_ARCH_PATH = 'model/model_architecture.png'


def create_model(num_classes, input_shape=(224, 224, 3)):
    # Define the model using the Functional API
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

    base_model = tf.keras.Model(inputs, outputs)

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.2,
            final_sparsity=0.8,
            begin_step=2000,
            end_step=10000
        )
    }

    model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    return model


def train_model(train_generator, val_generator, class_names):
    num_classes = len(class_names)
    model = create_model(num_classes)

    # Uncomment to plot the model architecture
    # plot_model(
    #     model,
    #     to_file=MODEL_ARCH_PATH,
    #     show_shapes=True,
    #     show_layer_names=True
    # )

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
        'best_model.h5',
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

    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs')
    ]

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50,
        callbacks=[checkpoint, early_stopping, reduce_lr] + pruning_callbacks
    )

    final_model = tfmot.sparsity.keras.strip_pruning(model)
    final_model.save(MODEL_PATH)
    return final_model


if __name__ == "__main__":
    train_gen, val_gen, class_names = preprocess_data(DATA_PATH, seed=42)
    np.save(CLASS_NAME_PATH, class_names)

    model = train_model(train_gen, val_gen, class_names)