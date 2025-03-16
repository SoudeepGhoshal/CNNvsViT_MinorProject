import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def preprocess_data(data_dir, img_size=(224, 224), test_size=0.2, batch_size=32, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=test_size
    )

    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=test_size)

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=seed
    )

    val_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=seed
    )

    class_names = list(train_generator.class_indices.keys())
    return train_generator, val_generator, class_names

def main():
    pass

if __name__ == "__main__":
    main()