import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from model import build_model

def main(args):

    train_ds = image_dataset_from_directory(
        os.path.join(args.data_dir, 'train'),
        batch_size=args.batch_size,
        image_size=(224, 224),
        label_mode='binary'
    )

    val_ds = image_dataset_from_directory(
        os.path.join(args.data_dir, 'val'),
        batch_size=args.batch_size,
        image_size=(224, 224),
        label_mode='binary'
    )

    model = build_model()

    os.makedirs("models", exist_ok=True)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        save_best_only=True,
        monitor="val_accuracy"
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
