import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data_dir, img_size=(224, 224), batch_size=32, seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Step 1: Load all images and their class labels
    data = []
    targets = []
    class_labels = sorted(os.listdir(data_dir))
    class_to_idx = {label: idx for idx, label in enumerate(class_labels)}

    for class_label in class_labels:
        class_dir = os.path.join(data_dir, class_label)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if os.path.isfile(img_path):
                data.append(img_path)
                targets.append(class_to_idx[class_label])

    data = np.array(data)
    targets = np.array(targets)
    total_samples = len(data)
    print(f"Loaded {total_samples} images belonging to {len(class_labels)} classes.")

    assert len(class_labels) == 67, f"Expected 67 classes, found {len(class_labels)}"

    # Step 2: First split - 80% train, 20% temp
    train_data, temp_data, train_targets, temp_targets = train_test_split(
        data, targets, train_size=0.80, stratify=targets, random_state=seed
    )

    # Step 3: Second split - 20% temp into 70% test (14% total) and 30% val (6% total)
    test_data, val_data, test_targets, val_targets = train_test_split(
        temp_data, temp_targets, test_size=0.30, stratify=temp_targets, random_state=seed
    )

    # Verify split sizes
    train_size = len(train_data)
    test_size = len(test_data)
    val_size = len(val_data)
    print(f"Train samples: {train_size} ({train_size/total_samples*100:.1f}%)")
    print(f"Test samples: {test_size} ({test_size/total_samples*100:.1f}%)")
    print(f"Val samples: {val_size} ({val_size/total_samples*100:.1f}%)")

    # Step 4: Create data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': train_data, 'class': train_targets.astype(str)}),
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )

    test_generator = test_val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': test_data, 'class': test_targets.astype(str)}),
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed
    )

    val_generator = test_val_datagen.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': val_data, 'class': val_targets.astype(str)}),
        x_col='filename',
        y_col='class',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        seed=seed
    )

    # Calculate steps per epoch
    train_steps = int(np.ceil(train_size / batch_size))
    test_steps = int(np.ceil(test_size / batch_size))
    val_steps = int(np.ceil(val_size / batch_size))

    # Wrap generators in tf.data.Dataset to enable repeating
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *img_size, 3], [None, len(class_labels)])
    ).repeat()

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *img_size, 3], [None, len(class_labels)])
    ).take(test_steps)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: val_generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *img_size, 3], [None, len(class_labels)])
    ).take(val_steps)

    class_names = class_labels

    return (train_dataset, test_dataset, val_dataset,
            train_steps, test_steps, val_steps,
            class_names)

if __name__ == "__main__":
    DATA_PATH = 'data/archive/indoorCVPR_09/Images'
    train_gen, test_gen, val_gen, train_steps, test_steps, val_steps, class_names = preprocess_data(DATA_PATH)