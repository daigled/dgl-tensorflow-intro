import keras.models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import urllib.request
import zipfile
import os
import progressbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def fetch_data(target_url, filename):
    target_directory = f"./tmp/"
    filename = filename + ".zip"

    try:
        zip_ref = zipfile.ZipFile(filename, "r")
    except FileNotFoundError:
        print(f"no local copy found, fetching data from {target_url}")
        urllib.request.urlretrieve(target_url, filename, show_progress)
        zip_ref = zipfile.ZipFile(filename, "r")

    zip_ref.extractall(target_directory)
    zip_ref.close()
    print(f"data successfully retrieved, and wrote {filename} to {target_directory}")


def show_sample_data():
    # Setup Local Context
    rock_dir = os.path.join("./tmp/rps/rock")
    paper_dir = os.path.join("./tmp/rps/paper")
    scissors_dir = os.path.join("./tmp/rps/scissors")

    print("total training rock images:", len(os.listdir(rock_dir)))
    print("total training paper images:", len(os.listdir(paper_dir)))
    print("total training scissors images:", len(os.listdir(scissors_dir)))

    rock_files = os.listdir(rock_dir)
    print(rock_files[:10])

    paper_files = os.listdir(paper_dir)
    print(paper_files[:10])

    scissors_files = os.listdir(scissors_dir)
    print(scissors_files[:10])

    # Visualize Some Stuff
    pic_index = 2

    next_rock = [
        os.path.join(rock_dir, fname) for fname in rock_files[pic_index - 2 : pic_index]
    ]
    next_paper = [
        os.path.join(paper_dir, fname)
        for fname in paper_files[pic_index - 2 : pic_index]
    ]
    next_scissors = [
        os.path.join(scissors_dir, fname)
        for fname in scissors_files[pic_index - 2 : pic_index]
    ]

    for i, img_path in enumerate(next_rock + next_paper + next_scissors):
        # print(img_path)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        plt.axis("Off")
        plt.show()


def getImageGenerator(type=None):
    if type == "IMG_AUG":
        return ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )

    return ImageDataGenerator(rescale=1.0 / 255)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epochs, logs={}):
        if logs.get("acc") > 0.95:
            print("\nReached 95% accuracy, training complete")
            self.model.stop_training = True


def main():
    callbacks = myCallback()

    # Fetch Training Data
    TRAINING_DATA_URL = "https://storage.googleapis.com/learning-datasets/rps.zip"
    TRAINING_DATA_FILENAME = "rps"
    fetch_data(TRAINING_DATA_URL, TRAINING_DATA_FILENAME)

    # Fetch Testing Data
    TESTING_DATA_URL = (
        "https://storage.googleapis.com/learning-datasets/rps-test-set.zip"
    )
    TESTING_DATA_FILENAME = "rps-test-set"
    fetch_data(TESTING_DATA_URL, TESTING_DATA_FILENAME)

    # show_sample_data()

    TRAINING_DIR = "./tmp/rps"
    training_datagen = getImageGenerator(type="IMG_AUG")

    VALIDATION_DIR = "./tmp/rps"
    validation_datagen = getImageGenerator()

    training_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        class_mode="categorical",  # NB: binary class_mode won't work here since we have more than 2 classes
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR, target_size=(150, 150), class_mode="categorical"
    )

    try:
        model = keras.models.load_model("rps.h5")
    except OSError:
        model = tf.keras.models.Sequential(
            [
                # Note the input shape is the desired size of the image 150x150 with 3 bytes color
                # This is the first convolution
                tf.keras.layers.Conv2D(
                    64, (3, 3), activation="relu", input_shape=(150, 150, 3)
                ),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The second convolution
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The third convolution
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # The fourth convolution
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # Flatten the results to feed into a DNN
                tf.keras.layers.Flatten(),
                # This Dropout layer will at random remove 50% of the input neurons
                tf.keras.layers.Dropout(0.5),
                # 512 neuron hidden layer
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])

    history = model.fit(
        training_generator,
        epochs=25,
        validation_data=validation_generator,
        verbose=1,
        callbacks=callbacks,
    )

    model.save("rps_test.h5")

    # Visualize Result
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(len(acc))

    plt.plot(epochs, acc, "r", label="Training accuracy")
    plt.plot(epochs, val_acc, "b", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend(loc=0)
    plt.figure()

    plt.show()


if __name__ == "__main__":
    main()
