import os
import glob
import json
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras.preprocessing import image_dataset_from_directory


def find_latest_run(models_dir="models"):
    runs = sorted(glob.glob(os.path.join(models_dir, "run_20260224_133643")))
    if not runs:
        raise FileNotFoundError(f"No run folders found in: {models_dir}/run_*")
    return runs[-1]


def load_run_config(run_dir):
    cfg_path = os.path.join(run_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config_used.yaml in run folder: {run_dir}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_split_ds(data_dir, split, img_size, batch_size):
    ds = image_dataset_from_directory(
        os.path.join(data_dir, split),
        batch_size=batch_size,
        image_size=img_size,
        label_mode="binary",
        shuffle=False
    )
    class_names = ds.class_names
    ds = ds.ignore_errors().prefetch(tf.data.AUTOTUNE)
    return ds, class_names


def get_probs_and_labels(ds, infer_fn):
    y_true, y_prob = [], []
    for x, y in ds:
        out = infer_fn(x)  # {"prob": tensor}
        probs = out["prob"].numpy().reshape(-1)
        y_true.append(y.numpy().reshape(-1))
        y_prob.append(probs)

    y_true = np.concatenate(y_true, axis=0).astype(int)
    y_prob = np.concatenate(y_prob, axis=0).astype(float)
    return y_true, y_prob


def pick_best_threshold(y_true, y_prob):
    best = {"thr": 0.5, "f1": -1.0}
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred)  # positive class = 1 (real)
        if f1 > best["f1"]:
            best = {"thr": float(thr), "f1": float(f1)}
    return best["thr"], best["f1"]


def compute_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report": classification_report(
            y_true, y_pred, target_names=["fake(0)", "real(1)"], digits=4
        ),
    }


def save_confusion_matrix_png(cm, out_path, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["fake(0)", "real(1)"])
    plt.yticks([0, 1], ["fake(0)", "real(1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    run_dir = find_latest_run("models")
    config = load_run_config(run_dir)

    data_dir = config["data"]["data_dir"]
    img_size_int = int(config["data"]["img_size"])
    img_size = (img_size_int, img_size_int)
    batch_size = int(config["data"]["batch_size"])

    print("Evaluating run:", run_dir)
    print("Using data_dir:", data_dir)
    print("Using img_size:", img_size)
    print("Using batch_size:", batch_size)

    savedmodel_dir = os.path.join(run_dir, "best_model_savedmodel")
    if not os.path.exists(savedmodel_dir):
        raise FileNotFoundError(f"Missing SavedModel at: {savedmodel_dir}")

    loaded = tf.saved_model.load(savedmodel_dir)
    infer = loaded.signatures["serving_default"]

    # ----- VAL -----
    val_ds, class_names = load_split_ds(data_dir, "val", img_size, batch_size)
    print("Class names:", class_names)

    y_true_val, y_prob_val = get_probs_and_labels(val_ds, infer)
    thr, best_f1 = pick_best_threshold(y_true_val, y_prob_val)
    val_metrics = compute_metrics(y_true_val, y_prob_val, thr)

    print(f"\nVAL AUC: {val_metrics['auc']:.4f}")
    print(f"Best threshold on VAL: {thr:.2f} (F1={best_f1:.4f})")
    print(val_metrics["report"])

    # Save VAL artifacts
    with open(os.path.join(run_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(os.path.join(run_dir, "val_report.txt"), "w") as f:
        f.write(val_metrics["report"])
    save_confusion_matrix_png(
        np.array(val_metrics["confusion_matrix"]),
        os.path.join(run_dir, "val_confusion_matrix.png"),
        "VAL Confusion Matrix"
    )

    # ----- TEST (if exists) -----
    test_dir = os.path.join(data_dir, "test")
    if os.path.exists(test_dir):
        test_ds, _ = load_split_ds(data_dir, "test", img_size, batch_size)
        y_true_test, y_prob_test = get_probs_and_labels(test_ds, infer)
        # IMPORTANT: use the threshold chosen on VAL (good practice)
        test_metrics = compute_metrics(y_true_test, y_prob_test, thr)

        print(f"\nTEST AUC: {test_metrics['auc']:.4f}")
        print(f"Using VAL threshold: {thr:.2f}")
        print(test_metrics["report"])

        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        with open(os.path.join(run_dir, "test_report.txt"), "w") as f:
            f.write(test_metrics["report"])
        save_confusion_matrix_png(
            np.array(test_metrics["confusion_matrix"]),
            os.path.join(run_dir, "test_confusion_matrix.png"),
            "TEST Confusion Matrix"
        )
    else:
        print("\nNo test/ folder found. Skipping test evaluation.")

    print("\nSaved evaluation artifacts into:", run_dir)


if __name__ == "__main__":
    main()
