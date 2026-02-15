import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing import image_dataset_from_directory

def main():

    test_ds = image_dataset_from_directory(
        "data/test",
        batch_size=32,
        image_size=(224, 224),
        label_mode='binary',
        shuffle=False
    )

    model = tf.keras.models.load_model("models/best_model.keras")

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = (model.predict(test_ds) > 0.5).astype(int).reshape(-1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()
