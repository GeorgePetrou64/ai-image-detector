import os
import random
import json
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def load_image(img_path, img_size):
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img).astype("float32")  # 0–255 (matches training)
    arr = np.expand_dims(arr, axis=0)
    return arr


def get_random_images(folder, n=5):
    files = []
    for f in os.listdir(folder):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            files.append(os.path.join(folder, f))
    return random.sample(files, min(n, len(files)))


def main():

    run_folder = input("Enter run folder path: ").strip()
    
    config_path = os.path.join(run_folder, "config_used.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError("config_used.yaml not found")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_dir = config["data"]["data_dir"]
    img_size_int = int(config["data"]["img_size"])
    img_size = (img_size_int, img_size_int)

    print(f"Using img_size: {img_size}")
    print(f"Using test folder: {data_dir}")

    # Load threshold from validation metrics if exists
    threshold = 0.5
    val_metrics_path = os.path.join(run_folder, "val_metrics.json")
    if os.path.exists(val_metrics_path):
        with open(val_metrics_path, "r") as f:
            metrics = json.load(f)
            threshold = float(metrics.get("threshold", 0.5))
    print(f"Using threshold: {threshold}")

    # Load SavedModel
    savedmodel_dir = os.path.join(run_folder, "best_model_savedmodel")
    loaded = tf.saved_model.load(savedmodel_dir)
    infer = loaded.signatures["serving_default"]

    # Get test folders
    fake_folder = os.path.join(data_dir, "test", "fake")
    real_folder = os.path.join(data_dir, "test", "real")

    fake_images = get_random_images(fake_folder, 5)
    real_images = get_random_images(real_folder, 5)

    samples = [(p, 0) for p in fake_images] + [(p, 1) for p in real_images]
    random.shuffle(samples)

    correct = 0

    print("\n=== Predictions ===\n")

    for path, true_label in samples:
        x = load_image(path, img_size)
        x_tf = tf.convert_to_tensor(x, dtype=tf.float32)

        out = infer(x_tf)
        prob_real = float(out["prob"].numpy().reshape(-1)[0])
        prob_fake = 1.0 - prob_real

        pred_label = 1 if prob_real >= threshold else 0

        if pred_label == true_label:
            correct += 1

        true_str = "REAL" if true_label == 1 else "FAKE"
        pred_str = "REAL" if pred_label == 1 else "FAKE"

        print(f"{os.path.basename(path)}")
        print(f"  True: {true_str}")
        print(f"  Prob REAL: {prob_real:.4f}")
        print(f"  Prob FAKE: {prob_fake:.4f}")
        print(f"  Pred: {pred_str}")
        print("-" * 40)

    print(f"\nAccuracy on 10-sample smoke test: {correct}/10")


if __name__ == "__main__":
    main()