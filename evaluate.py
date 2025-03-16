import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, val_generator):
    y_pred = model.predict(val_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_generator.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes,
                                target_names=val_generator.class_indices.keys()))

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=val_generator.class_indices.keys(),
                yticklabels=val_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    accuracy = np.mean(y_pred_classes == y_true)
    print(f"\nOverall Accuracy: {accuracy:.4f}")


def main():
    model = tf.keras.models.load_model('final_model.h5')
    from preprocess import preprocess_data
    _, val_generator, class_names = preprocess_data("images")
    evaluate_model(model, val_generator)


if __name__ == "__main__":
    main()