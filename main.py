import tensorflow as tf
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# There exists set of X values and a set of Y values with a constant relationship (Y = 2X - 1).
# Given these two sets and an arbitrary X value, determine the Y value.
def main():
    # Sequential is how we define the layers of the neural net.
    # Dense layers mean that all neurons in a layer are connected to all neurons in the next layer; "fully connected".
    # Dense layers are the most common layer type.
    # Here, we define a single dense layer comprised of 1 neuron.
    l0 = Dense(units=1, input_shape=[1])
    model = Sequential([l0])

    # Our one neuron learns a Weight (W) and Bias (B) such that Y = WX + B
    #   note the similarity of this function to the function which defines the relationship between X and Y
    #   (Y = WX + B) == (Y = 2X - 1) if W = 2 and B = -1

    # loss: Loss Function; that's how our model will gauge how good or bad its guess is.
    # optimizer: uses the feedback from the Loss Function to change the next guess.
    # sgd = Stochiastic Gradient.
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

    # the learning process
    # "Fit the X's to the Y's and try it 500 times"
    model.fit(xs, ys, epochs=500)

    # NB: the term predict here has to do with the amount of uncertainty in the returned value, it does not
    #   imply clairvoyance
    # NB: the model will probably not return the exact value, but a value very very close to the expected out for
    #   a given input of 10.0: this is because the loss is APPROACHING zero, but it will never reach zero
    print(model.predict([10.0]))

    # output should be very very close to [2, -1]
    print("Here is what I learned: {}".format(l0.get_weights()))


if __name__ == '__main__':
    main()

