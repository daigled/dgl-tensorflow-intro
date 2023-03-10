import tensorflow as tf
import urllib.request
import zipfile
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# Computer Vision: Image Augmentation
# Training Data: Horses or Humans
# Helpful: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/horses_or_humans.py


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('acc') > 0.95:
            print("\nReached 95% accuracy, training complete")
            self.model.stop_training = True


def get_training_data(target_directory):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"

    filename = "horse-or-human.zip"
    urllib.request.urlretrieve(url, filename)

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(target_directory)
    zip_ref.close()
    print('Training Data succesfully retrieved')
    

def get_validation_data(target_directory):
    url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    filename = "validation-horse-or-human.zip"

    urllib.request.urlretrieve(url, filename)

    validation_zip_ref = zipfile.ZipFile(filename, 'r')
    validation_zip_ref.extractall(target_directory)
    validation_zip_ref.close()
    print('Validation Data succesfully retrieved')


def test_images(model, kernel=(300, 300)):
    # Test Model with manually selected images
    image_path = 'assets/horses-and-humans-photos/'

    for item in os.listdir(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path+item, target_size=kernel)

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


def image_augmentation():
    callbacks = myCallback()
    
    # Fetch Training Data
    training_dir = "horse-or-human/training"
    get_training_data(training_dir)

    # Fetch Validation Data
    validation_dir = "horse-or-human/validation"
    get_validation_data(validation_dir)

    # Labelling with ImageDataGenerator
    # first, all images will be rescaled by 1./255
    # train_datagen = ImageDataGenerator(rescale=1/255)
    validation_datagen = ImageDataGenerator(rescale=1/255)

    # Let's use Image Augmentation on our ImageDataGenerator to prevent the model from having a bias
    #   towards horses.
    # Note: this will make training take longer but can help prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,      # rotate each image randomly up to 40deg left or right
        width_shift_range=0.2,  # translate each image up to 20% along X axis
        height_shift_range=0.2, # translate each image up to 20% along Y axis
        shear_range=0.2,        # shear each image up to 20%
        zoom_range=0.2,         # zoom each image up to 20%
        horizontal_flip=True,   # randomly flip the image horizontally or vertically
        fill_mode='nearest'     # fill any missing pixels after a move or shear with the nearest neighbors
    )

    # target_size is hyperparameter describing the size of images we want to handle
    # we use binary classification since there are 2 types of images we're labelling: horse and human
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        class_mode='binary'
    )

    # Factors impacting model design:
    #   1. Image size is now 300x300
    #   2. Images are now in color, not greyscale
    #   3. 2 image types instead of 10 like in Fashion MNIST, so we can use a binary classifier
    model = tf.keras.models.Sequential([
        # Convolutional Layer I
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        layers.MaxPooling2D(2, 2),
        # Convolutional Layer II
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # Convolutional Layer III
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # Convolutional Layer IV
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # Convolutional Layer V
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        # Flattened Input Layer
        layers.Flatten(),
        # Hidden Layer
        layers.Dense(512, activation='relu'),
        # Output Layer
        layers.Dense(1, activation='sigmoid')
    ])

    # Let's take a look at the layers of our model so far
    model.summary()

    # Model Training
    # RMSprop: Root Mean Square Propagation; takes learning rate (lr) parameter
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['acc']
    )

    # Fit Model against Training Data and use Validation Generator
    model.fit(
        train_generator,
        epochs=15,
        callbacks=[callbacks],
        validation_data=validation_generator
    )

    test_images(model=model)


def transfer_learning():
    callbacks = myCallback()

    # Call in pre-existing layers
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"

    urllib.request.urlretrieve(weights_url, weights_file)

    # Load Existing Layers into model
    pre_trained_model = InceptionV3(
        input_shape=(150, 150, 3),
        include_top=False,
        weights=None
    )

    pre_trained_model.load_weights(weights_file)

    # Summarize Pre-Trained Model
    pre_trained_model.summary()

    # Capture the last layer of output that we're interested in from the pre-existing model
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape:  ', last_layer.output_shape)
    last_output = last_layer.output

    # Build the layers of our own model underneath the pre-existing layers
    # Flatten captured output layer to 1 dimension
    my_model_layers = layers.Flatten()(last_output)
    # Now let's add a fully connected layer with 1024 hidden neurons and ReLU activation
    my_model_layers = layers.Dense(1024, activation='relu')(my_model_layers)
    # Let's add a final sigmoid layer for classification
    my_model_layers = layers.Dense(1, activation='sigmoid')(my_model_layers)

    model = Model(pre_trained_model.input, my_model_layers)

    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['acc'])

    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    train_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,  # rotate each image randomly up to 40deg left or right
        width_shift_range=0.2,  # translate each image up to 20% along X axis
        height_shift_range=0.2,  # translate each image up to 20% along Y axis
        shear_range=0.2,  # shear each image up to 20%
        zoom_range=0.2,  # zoom each image up to 20%
        horizontal_flip=True,  # randomly flip the image horizontally or vertically
        fill_mode='nearest'  # fill any missing pixels after a move or shear with the nearest neighbors
    )

    training_dir = "horse-or-human/training"
    validation_dir = "horse-or-human/validation"

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(150, 150),
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        class_mode='binary'
    )

    model.fit(
        train_generator,
        epochs=15,
        callbacks=[callbacks],
        validation_data=validation_generator
    )

    test_images(model=model, kernel=(150, 150))


if __name__ == '__main__':
    # image_augmentation()
    transfer_learning()
