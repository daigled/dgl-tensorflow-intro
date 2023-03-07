import tensorflow as tf
import urllib.request
import zipfile
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# Computer Vision: Image Augmentation
# Training Data: Horses or Humans
# Helpful: https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/horses_or_humans.py

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy, training complete")
            self.model.stop_training = True


def main():
    callbacks = myCallback()
    # Fetch Training Data
    url = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"

    filename = "horse-or-human.zip"
    training_dir = "horse-or-human/training"
    urllib.request.urlretrieve(url, filename)

    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(training_dir)
    zip_ref.close()

    # Fetch Validation Data
    validation_url = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    validation_filename = "validation-horse-or-human.zip"
    validation_dir = "horse-or-human/validation"

    urllib.request.urlretrieve(validation_url, validation_filename)

    validation_zip_ref = zipfile.ZipFile(validation_filename, 'r')
    validation_zip_ref.extractall(validation_dir)
    validation_zip_ref.close()

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
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Convolutional Layer II
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Convolutional Layer III
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Convolutional Layer IV
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Convolutional Layer V
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flattened Input Layer
        tf.keras.layers.Flatten(),
        # Hidden Layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Output Layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Let's take a look at the layers of our model so far
    model.summary()

    # Model Training
    # RMSprop: Root Mean Square Propagation; takes learning rate (lr) parameter
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Fit Model against Training Data and use Validation Generator
    history = model.fit(
        train_generator,
        epochs=15,
        callbacks=[callbacks],
        validation_data=validation_generator
    )

    # Test Model with manually selected images
    image_path = 'assets/horses-and-humans-photos/'

    for item in os.listdir(image_path):
        img = tf.keras.preprocessing.image.load_img(image_path+item, target_size=(300, 300))

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




if __name__ == '__main__':
    main()

