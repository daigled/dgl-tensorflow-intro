import tensorflow as tf
import urllib.request
import os
import numpy as np

# Transfer Learning - related libs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# Computer Vision: Transfer Learning
# Training Data: Horses or Humans
# Helpful: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/horses_or_humans.py


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get("acc") > 0.95:
            print("\nReached 95% accuracy, training complete")
            self.model.stop_training = True


def test_images(model):
    # Test Model with manually selected images
    image_path = "assets/horses-and-humans-photos/"

    for item in os.listdir(image_path):
        img = tf.keras.preprocessing.image.load_img(
            image_path + item, target_size=(150, 150)
        )

        x = tf.keras.preprocessing.image.img_to_array(img)

        x = np.expand_dims(x, axis=0)

        image_tensor = np.vstack([x])

        classes = model.predict(image_tensor)

        print("\nPrediction Time:")
        print(classes)
        print(classes[0])

        if classes[0] > 0.5:
            print(item + " is a human")
        else:
            print(item + " is a horse")


def main():
    callbacks = myCallback()

    # Setup
    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,  # rotate each image randomly up to 40deg left or right
        width_shift_range=0.2,  # translate each image up to 20% along X axis
        height_shift_range=0.2,  # translate each image up to 20% along Y axis
        shear_range=0.2,  # shear each image up to 20%
        zoom_range=0.2,  # zoom each image up to 20%
        horizontal_flip=True,  # randomly flip the image horizontally or vertically
        fill_mode="nearest",  # fill any missing pixels after a move or shear with the nearest neighbors
    )

    training_dir = "horse-or-human/training"
    validation_dir = "horse-or-human/validation"

    train_generator = train_datagen.flow_from_directory(
        training_dir, target_size=(150, 150), class_mode="binary"
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, target_size=(150, 150), class_mode="binary"
    )

    # Call in pre-existing layers
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"

    urllib.request.urlretrieve(weights_url, weights_file)

    # Load Existing Layers into model
    pre_trained_model = InceptionV3(
        input_shape=(150, 150, 3), include_top=False, weights=None
    )

    pre_trained_model.load_weights(weights_file)

    # Summarize Pre-Trained Model
    pre_trained_model.summary()

    # Freeze Pre-Trained Model Layers
    for layer in pre_trained_model.layers:
        layer.trainable = False

    # Capture the last layer of output that we're interested in from the pre-existing model
    last_layer = pre_trained_model.get_layer("mixed7")
    print("last layer output shape:  ", last_layer.output_shape)
    last_output = last_layer.output

    # Build the layers of our own model underneath the pre-existing layers
    # Flatten captured output layer to 1 dimension
    my_model_layers = layers.Flatten()(last_output)
    # Now let's add a fully connected layer with 1024 hidden neurons and ReLU activation
    my_model_layers = layers.Dense(1024, activation="relu")(my_model_layers)
    # Let's add a final sigmoid layer for classification
    my_model_layers = layers.Dense(1, activation="sigmoid")(my_model_layers)

    model = Model(pre_trained_model.input, my_model_layers)

    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["acc"],
    )

    model.fit(
        train_generator,
        epochs=15,
        callbacks=[callbacks],
        validation_data=validation_generator,
    )

    test_images(model=model)


if __name__ == "__main__":
    main()
