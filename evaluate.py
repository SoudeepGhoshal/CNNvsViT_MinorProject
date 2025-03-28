import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, r2_score, roc_curve, auc
from itertools import cycle
import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = 'model/with_pruning/cnn_pruning.keras'
HISTORY_PATH = 'model/with_pruning/training_history.json'
DATA_PATH = 'data/archive/indoorCVPR_09/Images'
EVAL_PATH = 'eval/with_pruning'

if not os.path.exists(EVAL_PATH):
    try:
        os.makedirs(EVAL_PATH, exist_ok=True)
        logger.info(f"Created evaluation directory: {EVAL_PATH}")
    except OSError as e:
        logger.error(f"Failed to create evaluation directory {EVAL_PATH}: {e}")
        raise


def plot_roc_curves(y_true, y_pred, class_names, save_path):
    """Plot and save ROC curves for each class and micro-average with improved legend placement."""
    try:
        # Convert true labels to one-hot encoding
        y_true_one_hot = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
        n_classes = len(class_names)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Calculate ROC for each class (one-vs-rest)
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_one_hot[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate micro-average ROC curve
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curves
        plt.figure(figsize=(12, 8))  # Increased figure size for better visibility
        colors = cycle(['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'gray', 'brown', 'cyan'])

        # Plot each class's ROC curve
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', lw=2, linestyle='--',
                 label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})')

        # Plot diagonal line (random guessing)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.grid(True)

        # Adjust legend placement: move it further to the right and split into columns
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5),
                   ncol=2,  # Split legend into 2 columns
                   borderaxespad=0.,
                   fontsize='small',  # Reduce font size to fit more entries
                   frameon=False)  # Remove legend border for cleaner look

        # Save the plot
        plt.savefig(save_path, bbox_inches='tight')  # Ensure the legend is included in the saved image
        logger.info(f"ROC curves plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Failed to plot ROC curves: {e}")
    finally:
        plt.close()

def plot_classification_counts(y_true, y_pred_classes, class_names, save_path):
    """Plot and save a bar chart of true vs predicted class counts."""
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred_classes, minlength=len(class_names))

    plt.figure(figsize=(15, 6))
    x = np.arange(len(class_names))
    width = 0.35

    plt.bar(x - width/2, true_counts, width, label='True', color='blue')
    plt.bar(x + width/2, pred_counts, width, label='Predicted', color='orange')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('True vs Predicted Class Counts')
    plt.xticks(x, class_names, rotation=90)
    plt.legend()
    plt.tight_layout()

    try:
        plt.savefig(save_path)
        logger.info(f"Classification counts plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save classification counts plot: {e}")
    finally:
        plt.close()

def plot_training_history(history_path, save_path):
    """Load training history and plot accuracy and loss in a single image."""
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load training history from {history_path}: {e}")
        raise

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Model Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    try:
        plt.savefig(save_path)
        logger.info(f"Training history plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save training history plot: {e}")
    finally:
        plt.close()


def evaluate_model(model, test_generator, class_names):
    """Evaluate the model on the test set and save results."""
    # Get predictions and true labels
    try:
        # For datasets, we need to iterate through all batches
        y_pred = []
        y_true = []

        # Reset the dataset iterator if needed
        for batch_x, batch_y in test_generator:
            predictions = model.predict(batch_x, verbose=1)
            y_pred.append(predictions)
            y_true.append(batch_y)

        # Concatenate all batches
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        # Convert predictions and true labels to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true

    except Exception as e:
        logger.error(f"Failed to generate predictions: {e}")
        raise

    # Classification Report
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    logger.info("\nClassification Report:\n" + report)
    report_path = os.path.join(EVAL_PATH, 'classification_report.txt')
    try:
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Classification report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save classification report: {e}")

    # Confusion Matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(25, 20))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(EVAL_PATH, 'confusion_matrix.png')
    try:
        plt.savefig(cm_path)
        logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix: {e}")
    finally:
        plt.close()

    # Classification Counts
    counts_path = os.path.join(EVAL_PATH, 'classification_counts.png')
    plot_classification_counts(y_true_classes, y_pred_classes, class_names, counts_path)

    # ROC Curves
    roc_path = os.path.join(EVAL_PATH, 'roc_curves.png')
    plot_roc_curves(y_true_classes, y_pred, class_names, roc_path)

    # Calculate metrics
    accuracy = np.mean(y_pred_classes == y_true_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    y_true_one_hot = tf.keras.utils.to_categorical(y_true_classes, num_classes=len(class_names))
    r2 = r2_score(y_true_one_hot, y_pred)
    n, p = len(y_true_classes), len(class_names)
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    # Log metrics
    metrics = (
        f"\nModel Evaluation Metrics:\n"
        f"Overall Accuracy: {accuracy:.4f}\n"
        f"Overall F1 Score (weighted): {f1:.4f}\n"
        f"Overall Recall (weighted): {recall:.4f}\n"
        f"R2 Score: {r2:.4f}\n"
        f"Adjusted R2 Score: {adjusted_r2:.4f}"
    )
    logger.info(metrics)

def main():
    # Load model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {e}")
        raise

    # Load test data using updated preprocess function
    from preprocess import preprocess_data
    try:
        (train_gen, test_gen, val_gen,
         train_steps, test_steps, val_steps,
         class_names) = preprocess_data(DATA_PATH, seed=42)
        logger.info("Test data loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise

    # Evaluate on test set
    evaluate_model(model, test_gen, class_names)

    # Plot and save training history
    history_plot_path = os.path.join(EVAL_PATH, 'training_history_plot.png')
    plot_training_history(HISTORY_PATH, history_plot_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise