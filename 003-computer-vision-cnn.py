import tensorflow as tf

# Computer Vision: CNN
# Training Data: Fashion MNIST
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get("accuracy") > 0.95:
            print("\nReached 95% accuracy, training complete")
            self.model.stop_training = True


def main():
    # Initialization
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()

    callbacks = myCallback()

    # Normalization
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    # Model Declaration
    # Note the introduction of Convolutional and Pooling Layers
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
            # Debugging note: if we use relu as the activation function for the output layer instead of softmax,
            #   accuracy never reaches a value higher than .10 - why?
        ]
    )

    # Give a summary of the shape of the model so far
    model.summary()

    # Model Training
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


if __name__ == "__main__":
    main()
