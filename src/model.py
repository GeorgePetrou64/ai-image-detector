import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_model(img_size=(224, 224), dropout=0.2):
    """
    Builds the model architecture ONLY (no compile).
    Compile happens in train.py so we can easily switch LR and fine-tune stage.
    """
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="augmentation")

    base_model = EfficientNetB0(
        input_shape=img_size + (3,),
        include_top=False,
        weights="imagenet"
    )
    # We'll control trainable layers from train.py
    base_model.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)  # stage 1 (stable BN)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="ai_image_detector")
    return model


def find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """
    Find the EfficientNet backbone submodel inside the Keras model.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name.lower().startswith("efficientnet"):
            return layer
    raise ValueError("EfficientNet backbone not found inside the model.")