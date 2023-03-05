import tensorflow as tf
data = tf.keras.datasets.fashion_mnist


# Computer Vision
# Training Data: Fashion MNIST
def main():
    (training_images, training_labels), (test_images, test_labels) = data.load_data()

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
        keras.layers.Flatten(input_shape=(28, 28)),
        # After the input layer we specify a hidden layer. The number of neurons in a "hidden" layer is arbitrary.
        #   More neurons means it will run slower because it has to learn more parameters.
        #   More neurons could lead to a NN which is great a recognizing training data,
        #   but sucks at recognizing new input ("overfitting"). Key idea is to have enough neurons to recognize key
        #   parameters of the dataset. ("Hyperparameter tuning")
        # Hyperparameters are values used to control training, Parameters are the internal values of the neurons
        #   that get trained.
        keras.layers.Dense(128, activation=tf.nn.relu),
        # After the hidden layer, we declare our output layer. It has 10 neurons because there are 10 classes of images
        #   for us to choose from in the Fashion MNIST dataset. Each of these neurons will end up with probability
        #   values. We will choose the neuron with the highest value. We could loop over them ourselves, but instead
        #   we use the softmax activation function which will do that for us.
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Why a different loss function here than the one we used in 001? Rather than determining a single value, in this
    #   case we're making a categorical selection.
    # The adam optimizer is a faster, more efficient sgd
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=5)

    model.evaluate(test_images, test_labels)


if __name__ == '__main__':
    main()

