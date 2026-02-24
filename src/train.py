import os
import json
import yaml
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

from model import build_model, find_backbone


def export_savedmodel_for_serving(model: tf.keras.Model, export_dir: str, img_size: tuple):
    """
    Export TF SavedModel using tf.saved_model.save (bypasses Keras JSON serialization).
    """
    dummy = tf.zeros((1, img_size[0], img_size[1], 3), dtype=tf.float32)
    _ = model(dummy, training=False)

    @tf.function(input_signature=[tf.TensorSpec([None, img_size[0], img_size[1], 3], tf.float32)])
    def serving_fn(x):
        y = model(x, training=False)
        return {"prob": y}

    if tf.io.gfile.exists(export_dir):
        tf.io.gfile.rmtree(export_dir)

    tf.saved_model.save(model, export_dir, signatures={"serving_default": serving_fn})


def set_fine_tune(backbone: tf.keras.Model, unfreeze_last_n: int):
    """
    Unfreeze last N layers of the backbone; freeze the rest.
    If unfreeze_last_n <= 0 => keep fully frozen.
    """
    backbone.trainable = True
    n = len(backbone.layers)

    if unfreeze_last_n <= 0:
        for l in backbone.layers:
            l.trainable = False
        return

    # Freeze all but last N layers
    cut = max(0, n - unfreeze_last_n)
    for i, l in enumerate(backbone.layers):
        l.trainable = i >= cut


def compute_balanced_class_weight(data_dir, class_names):
    """
    Balanced class weights: total / (num_classes * count_i)
    """
    counts = {name: 0 for name in class_names}
    for name in class_names:
        class_folder = os.path.join(data_dir, "train", name)
        if os.path.isdir(class_folder):
            counts[name] = sum(
                1 for fn in os.listdir(class_folder)
                if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            )

    total = sum(counts.values())
    if total <= 0 or any(counts[n] <= 0 for n in class_names):
        return None, counts

    class_weight = {
        i: total / (len(class_names) * counts[name])
        for i, name in enumerate(class_names)
    }
    return class_weight, counts


def compile_for_training(model: tf.keras.Model, lr: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def main():
    # =========================
    # LOAD CONFIG
    # =========================
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config.get("model", {})

    data_dir = data_cfg["data_dir"]
    img_size_int = int(data_cfg["img_size"])
    img_size = (img_size_int, img_size_int)
    batch_size = int(data_cfg["batch_size"])

    # Stage 1
    epochs = int(train_cfg["epochs"])
    lr1 = float(train_cfg["learning_rate"])

    # Stage 2 (optional)
    fine_tune = bool(train_cfg.get("fine_tune", False))
    lr2 = float(train_cfg.get("fine_tune_lr", 1e-5))
    unfreeze_last_n = int(train_cfg.get("fine_tune_layers", 0))  # interpreted as last N layers

    # Fast dev
    fast_dev_run = bool(train_cfg.get("fast_dev_run", False))
    fast_train_batches = int(train_cfg.get("fast_train_batches", 50))
    fast_val_batches = int(train_cfg.get("fast_val_batches", 20))

    dropout = float(model_cfg.get("dropout", 0.2))

    # Choose monitoring metric (AUC is better for detectors)
    monitor_metric = train_cfg.get("monitor", "val_auc")
    monitor_mode = "max" if ("auc" in monitor_metric or "acc" in monitor_metric) else "min"

    tf.keras.utils.set_random_seed(int(train_cfg.get("seed", 42)))
    AUTOTUNE = tf.data.AUTOTUNE

    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # =========================
    # RUN FOLDER
    # =========================
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("models", f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # =========================
    # DATASETS
    # =========================
    train_ds = image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        batch_size=batch_size,
        image_size=img_size,
        label_mode="binary",
        shuffle=True,
        seed=42
    )
    val_ds = image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        batch_size=batch_size,
        image_size=img_size,
        label_mode="binary",
        shuffle=False
    )

    class_names = train_ds.class_names
    print("Class names:", class_names)

    class_weight, counts = compute_balanced_class_weight(data_dir, class_names)
    print("Train counts:", counts)
    print("Class weight:", class_weight)

    train_ds = train_ds.ignore_errors()
    val_ds = val_ds.ignore_errors()

    os.makedirs("cache", exist_ok=True)
    train_ds = train_ds.cache(os.path.join("cache", "train")).prefetch(AUTOTUNE)
    val_ds = val_ds.cache(os.path.join("cache", "val")).prefetch(AUTOTUNE)

    if fast_dev_run:
        train_ds = train_ds.take(fast_train_batches)
        val_ds = val_ds.take(fast_val_batches)
        print(f"FAST DEV RUN: train_batches={fast_train_batches}, val_batches={fast_val_batches}")

    # =========================
    # BUILD MODEL (uncompiled)
    # =========================
    model = build_model(img_size=img_size, dropout=dropout)
    backbone = find_backbone(model)

    # =========================
    # CALLBACKS
    # =========================
    best_weights_path = os.path.join(run_dir, "best.weights.h5")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_weights_path,
        monitor=monitor_metric,
        mode=monitor_mode,
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        mode=monitor_mode,
        patience=int(train_cfg.get("early_stop_patience", 5)),
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
        patience=int(train_cfg.get("reduce_lr_patience", 2)),
        min_lr=float(train_cfg.get("min_lr", 1e-7)),
        verbose=1
    )

    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb]

    # =========================
    # STAGE 1: train head (freeze backbone)
    # =========================
    backbone.trainable = False
    compile_for_training(model, lr1)

    print("\n=== STAGE 1 (head) ===")
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight
    )

    # =========================
    # STAGE 2: fine-tune (optional)
    # =========================
    hist2 = None
    if fine_tune:
        # You can control how long stage2 runs via config too; default 3
        ft_epochs = int(train_cfg.get("fine_tune_epochs", 3))

        print(f"\n=== STAGE 2 (fine-tune) | unfreeze_last_n={unfreeze_last_n} | lr={lr2} | epochs={ft_epochs} ===")
        set_fine_tune(backbone, unfreeze_last_n)

        # Recompile with smaller LR
        compile_for_training(model, lr2)

        hist2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=ft_epochs,
            callbacks=callbacks,
            class_weight=class_weight
        )

    # Load best weights before export
    if os.path.exists(best_weights_path):
        model.load_weights(best_weights_path)

    # =========================
    # EXPORT SAVEDMODEL
    # =========================
    savedmodel_dir = os.path.join(run_dir, "best_model_savedmodel")
    export_savedmodel_for_serving(model, savedmodel_dir, img_size)
    print(f"\nExported SavedModel to: {savedmodel_dir}")

    # =========================
    # SAVE HISTORY + PLOT
    # =========================
    def merge_histories(h1, h2):
        merged = {}
        for k, v in (h1.history or {}).items():
            merged[k] = list(v)
        if h2 is not None:
            for k, v in (h2.history or {}).items():
                merged.setdefault(k, [])
                merged[k].extend(list(v))
        return merged

    merged = merge_histories(hist1, hist2)

    def to_float_list(xs):
        out = []
        for x in xs:
            try:
                out.append(float(x))
            except Exception:
                out.append(str(x))
        return out

    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({k: to_float_list(v) for k, v in merged.items()}, f, indent=2)

    plot_path = os.path.join(run_dir, "accuracy_curve.png")
    plt.plot(merged.get("accuracy", []), label="Train Accuracy")
    plt.plot(merged.get("val_accuracy", []), label="Val Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved history + plot into: {run_dir}")


if __name__ == "__main__":
    main()