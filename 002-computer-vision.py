import tensorflow as tf
data = tf.keras.datasets.fashion_mnist


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy, training complete")
            self.model.stop_training = True


# Computer Vision
# Training Data: Fashion MNIST
def main():
    (training_images, training_labels), (test_images, test_labels) = data.load_data()

    callbacks = myCallback()

    # Normalization
    # We know that the pixel values we want to work with are all 0 - 255
    # This notation allows us to perform the division operation on all values of an array
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    # Sequential: specify that this model has many layers

    # activation = Activation Function, the code which will execute on each neuron in the layer. relu = Rectified
    #   Linear Unit, which is a function which will return a value if it's greater than 0 (we don't want negative values
    #   passed to the next layer as this may impact summing function).
    model = tf.keras.models.Sequential([
        # Flatten: not a layer of neurons, but an input layer specification. Takes our "square" input of 28x28 pixels
        #   and "flattens" it into a 1D array
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        # After the input layer we specify a hidden layer. The number of neurons in a "hidden" layer is arbitrary.
        #   More neurons means it will run slower because it has to learn more parameters.
        #   More neurons could lead to a NN which is great a recognizing training data,
        #   but sucks at recognizing new input ("overfitting"). Key idea is to have enough neurons to recognize key
        #   parameters of the dataset. ("Hyperparameter tuning")
        # Hyperparameters are values used to control training, Parameters are the internal values of the neurons
        #   that get trained.
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        # After the hidden layer, we declare our output layer. It has 10 neurons because there are 10 classes of images
        #   for us to choose from in the Fashion MNIST dataset. Each of these neurons will end up with probability
        #   values. We will choose the neuron with the highest value. We could loop over them ourselves, but instead
        #   we use the softmax activation function which will do that for us.
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Why a different loss function here than the one we used in 001? Rather than determining a single value, in this
    #   case we're making a categorical selection.
    # The adam optimizer is a faster, more efficient sgd
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train our model to fit the training labels against the training images and do it 5 times
    # What happens when you set the epochs to 50? The model becomes Overfitted to the training data!
    #   96% accuracy on training, 89% accurate on test
    # We can add a callback to help the model figure out when it can stop training to avoid becoming specialized
    #   to a fault
    model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

    # See how well the model can apply test labels to test images now that it has been trained
    model.evaluate(test_images, test_labels)

    classifications = model.predict(test_images)
    print(classifications[0])  # This will show us the classification of the first test image
    print(test_labels[0])  # This will show us the first test_label

    # classifications[0]: [4.0213631e-06 4.1354369e-08 7.5193702e-07 4.1364012e-07 1.0317174e-07
    #  7.7535980e-04 4.8608655e-07 2.6824042e-02 1.1993355e-04 9.7227478e-01]
    #   Note that it's a collection of 10 values. Index 9 has the highest value, which means that the first test
    #       image will be classified with label 9. The value 9.7227478e-01 means that there's a 97% chance that 9 is
    #       the correct answer
    # test_labels[0]: 9. Our model got it right!

    # We can use this process of reviewing classifications and test labels to ensure that our model is "thinking"
    #   the right way


if __name__ == '__main__':
    main()

