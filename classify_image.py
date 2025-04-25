import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

MODEL_PATH = 'model/with_pruning/cnn_pruning.keras'
CLASS_NAME_PATH = 'model/with_pruning/class_names.npy'
IMG_PATH = 'inputs/image0.png' # Place the path of the image to classify

class GradCamPlusPlus:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )

    def compute_heatmap(self, img_array, pred_index=None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, img_path, alpha=0.4):
        img = load_img(img_path)
        img = img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        heatmap = tf.keras.preprocessing.image.array_to_img(heatmap[..., np.newaxis])
        heatmap = heatmap.resize((img.shape[1], img.shape[0]))
        heatmap = img_to_array(heatmap)
        heatmap = np.uint8(255 * heatmap / np.max(heatmap))
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(
            heatmap, data_format='channels_first'
        )
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
        return superimposed_img


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def predict_image(model, img_path, class_names, gradcam):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    heatmap = gradcam.compute_heatmap(img_array)
    superimposed_img = gradcam.overlay_heatmap(heatmap, img_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(load_img(img_path))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title(f"Grad-CAM++\nPredicted: {predicted_class} ({confidence:.4f})")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = np.load(CLASS_NAME_PATH, allow_pickle=True)
    gradcam = GradCamPlusPlus(model, 'last_conv')

    try:
        predicted_class, confidence = predict_image(model, IMG_PATH, class_names, gradcam)
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()