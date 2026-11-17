"""MobileNetV2 classifier and compilation helpers."""

import tensorflow as tf

from fasalmitra.config import IMAGE_SIZE, LEARNING_RATE


def build_model(class_count, image_size=IMAGE_SIZE, weights="imagenet",
                learning_rate=LEARNING_RATE):
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    backbone_weights = "imagenet" if weights == "imagenet" else None
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights=backbone_weights,
    )
    backbone.trainable = False
    features = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    features = tf.keras.layers.RandomFlip("horizontal_and_vertical")(features)
    features = tf.keras.layers.RandomRotation(0.1)(features)
    features = tf.keras.layers.RandomZoom(0.1)(features)
    features = backbone(features, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    features = tf.keras.layers.Dropout(0.2)(features)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(features)
    model = tf.keras.Model(inputs, outputs)
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
