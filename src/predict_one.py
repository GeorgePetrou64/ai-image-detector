import os
import sys
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def load_image(img_path, img_size):
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img).astype("float32")  # keep 0..255 (matches training)
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print('python src/predict_one.py <run_folder> <image_path> [threshold]')
        sys.exit(1)

    run_folder = sys.argv[1]
    img_path = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    # Load config_used.yaml for correct img_size
    config_path = os.path.join(run_folder, "config_used.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config_used.yaml not found in: {run_folder}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    img_size_int = int(config["data"]["img_size"])
    img_size = (img_size_int, img_size_int)
    print(f"Using img_size: {img_size}")

    # Load SavedModel (reliable in your TF/Keras setup)
    savedmodel_dir = os.path.join(run_folder, "best_model_savedmodel")
    if not os.path.exists(savedmodel_dir):
        raise FileNotFoundError(f"best_model_savedmodel not found in: {run_folder}")

    loaded = tf.saved_model.load(savedmodel_dir)
    infer = loaded.signatures["serving_default"]

    # Prepare image
    x = load_image(img_path, img_size)
    x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

    print("Input stats:",
      "min", float(tf.reduce_min(x_tf)),
      "max", float(tf.reduce_max(x_tf)),
      "mean", float(tf.reduce_mean(x_tf)))
    # Predict
    out = infer(x_tf)  # {"prob": ...}
    prob_real = float(out["prob"].numpy().reshape(-1)[0])
    prob_fake = 1.0 - prob_real

    print(f"\nProb REAL (1): {prob_real:.4f}")
    print(f"Prob FAKE (0): {prob_fake:.4f}")

    pred = "REAL" if prob_real >= threshold else "FAKE"
    print(f"Prediction: {pred} (threshold={threshold})")


if __name__ == "__main__":
    main()