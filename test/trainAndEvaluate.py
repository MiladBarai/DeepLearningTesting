# Import List
from numpy.core._multiarray_umath import ndarray
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.datasets import fashion_mnist as mnist
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python import keras
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import math
import numpy
import os
import time

# Category number to Category name list
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
augment_data = True


def train_or_load_network(network, training_images, training_lables):
    print("\n\nTRAIN OR LOAD --------------------------")

    # Refactoring network name since some characters where not allowed as TF name
    network_name = network.name
    network_name = network_name.replace("..", "[", 1)
    network_name = network_name.replace("_", "=")
    network_name = network_name.replace("..", ",")
    network_name = network_name + "]"

    folder = "../resources/Deep Neural Network Training/" + network_name + "/"

    if not os.path.exists(folder):
        print("Folder: " + folder + "\nDoes not exist, Running Training Mode")
        # Index out folds
        kfolds = 10
        fold_indexies = KFold(kfolds).split(training_images)

        print("\n" + str(kfolds) + "-Fold Cross-Validation initializing:")
        # Run training and evaluation for each fold
        for i, (train, test) in enumerate(fold_indexies):
            print("\nRunning Fold [" + str(i + 1) + " / " + str(kfolds) + "]\n")

            # reset the network weights between each fold
            network.load_weights("resetting_weights.h5")
            file_name = str(i + 1)

            # train, save best model and store training results in csv
            fit_and_evaluate(network, training_images[train], training_lables[train], training_images[test],
                             training_lables[test], folder, file_name)
    else:
        print("Restoring Models from: " + folder)
        print("Do restoring stuff ...")


def create_model(filter_size, dropout, padding_type, hidden_layers, use_max_pooling):
    print("\n\nCREATING NETWORK ------------------------")
    # Crafting a unique name for the nerual network depending on options selected
    network_name = "NN..FilterSize_" + str(filter_size) + "..DropOut_" + str(
        dropout) + "..Padding_" + padding_type + "..DataAugmentation_" + str(
        augment_data) + "..maxPooling_" + str(use_max_pooling)

    # Initialize sequencial NN
    network = models.Sequential(name=network_name)

    # Create an input layer for the neural network that is a Conv layer with 128 filters
    # with the input to be: 28x28 pixels and 1 channel
    network.add(layers.Conv2D(128, (filter_size, filter_size), padding=padding_type, activation='relu',
                              input_shape=(28, 28, 1)))

    # Create the amount of hidden input layers, with 2 conv layers and if options are selected
    # A max pooling and a drop out layer
    for i in range(hidden_layers):
        network.add(layers.Conv2D(128, (filter_size, filter_size), padding=padding_type, activation='relu'))
        network.add(layers.Conv2D(128, (filter_size, filter_size), padding=padding_type, activation='relu'))
        if use_max_pooling:
            network.add(layers.MaxPool2D(pool_size=(2, 2)))
        if dropout > 0:
            network.add(layers.Dropout(dropout))

    # Flatten the 2D structure
    network.add(layers.Flatten())

    # Set up a fully connected layer to the flattened structure
    network.add(layers.Dense(512, activation='relu'))

    # Set up a fully connected layer that will be the output prediction in %
    network.add(layers.Dense(10, activation='softmax'))

    # Compile model
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print a network summary
    network.summary()

    # Save the starting weights
    network.save_weights('resetting_weights.h5')

    return network


def get_one_img(index):
    (images, labels) = get_data()
    return images[index:index+1]


def get_data():
    print("\n\nLOADING DATA ---------------------------")
    # Load dataset
    (train_images, train_lables), (test_images, test_lables) = mnist.load_data()

    print("Training Images Length, before appending: " + str(len(train_images)))

    train_images = numpy.concatenate((train_images, test_images))
    train_lables = numpy.concatenate((train_lables, test_lables))

    print("Trainning Images Length, after appending: " + str(len(train_images)))

    # Transform from 70000, 28, 28 (images, x pixels, y pixels) to 70000, 28, 28, 1 (images, x pixels, y pixels, channels)
    train_images = train_images.reshape((70000, 28, 28, 1))
    print("Training images after reshape: " + str(train_images.shape))

    # Transform from 0-255 values for each pixel to 0-1 values
    train_images = train_images.astype('float32') / 255

    train_lables = to_categorical(train_lables)
    print("Base Data reshaped and loaded")

    return (train_images, train_lables)


def show_changed_images(sample_data):
    print("\n\nSHOWS SAMPLE ----------------------------")
    # Visualizing some samples out of the training data
    show_sample_size = len(sample_data)
    columns = 4
    rows = math.ceil(show_sample_size / 4)
    figure = plt.figure(figsize=(4, 4))
    for i in range(show_sample_size):
        figure.add_subplot(rows, columns, i + 1)
        plt.imshow(sample_data[i].reshape(28, 28), cmap=plt.get_cmap("gray"))
    plt.show()


def fit_and_evaluate(neural_net, train_img, train_lable, test_img, test_lable, folder, file_name):
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Base name for the file, (eg. folder/filename)
    base_file_naming = folder + file_name

    # Base name + file format
    checkpoint_file = base_file_naming + ".h5"
    csv_file = base_file_naming + ".csv"

    # Creating a model checkpointer which saves the best model out of the training epochs
    checkpoint_best = keras.callbacks.ModelCheckpoint(checkpoint_file, save_best_only=True)
    # Creating a csv logger to store the training results
    csv_logger = CSVLogger(csv_file)

    # Train neural network
    neural_net.fit(train_img, train_lable, epochs=50, batch_size=128, validation_data=(test_img, test_lable),
                   callbacks=[checkpoint_best, csv_logger])

    # Print saving information
    print("Saving best model with minimal loss: " + checkpoint_file)
    print("Saving model training progress: " + csv_file)


def augment_training_images(training_images, training_lables):
    # Init of the return variables
    img_result = training_images
    label_result = training_lables

    # Init place to store augmented images
    changed_images = numpy.ndarray(shape=(10000, 28, 28, 1))
    changed_lables = numpy.ndarray(shape=(10000, 10))

    print("\n\nAUGMENT DATA ----------------------------")
    # Define wanted augmentations and input training data for the augmentations
    data_generator = ImageDataGenerator(horizontal_flip=True, zoom_range=[0.75, 1.25])
    data_itterator = data_generator.flow(x=training_images, y=None, batch_size=128, shuffle=False)

    # Create a list for a viewable sample
    change_sample = []

    # Total images
    index = 0

    # Run through all the batches of augmented images
    for i in range(round(len(training_images) / 128)):
        img_data_batch = data_itterator.next()

        # Check each image in the batch
        for img_data in img_data_batch:
            # Exit condition. generate 10 000 extra images
            if index >= 10000:
                break

            # Add training data and lable to the changed lable arrays
            changed_images[index] = numpy.asarray(img_data)
            changed_lables[index] = training_lables[index]

            # Add the 20 fist images to the sample list
            if index < 20:
                change_sample.append(img_data)

            if len(changed_images) % 100 == 0:
                print("Augmentation Index reached: " + str(index + 1))

            # Increment the index of which img we are on at the moment
            index += 1

    # Merge original and augmented
    img_result = numpy.concatenate((img_result, changed_images))
    label_result = numpy.concatenate((label_result, changed_lables))

    print("Data augmentation completed, Zoom 75%-125% and Horizontal Flip")
    # Show the saved sample
    show_changed_images(change_sample)

    return img_result, label_result

def run():
    # Fetch formated data
    (training_images, training_labels) = get_data()

    # If Data augmentations has been selected then RANDOMLY, horizontaly_flip/mirror and zoom 25% enlargenment and normal size
    if augment_data:
        (training_images, training_labels) = augment_training_images(training_images, training_labels)

    print("Images: " + str(len(training_labels)))

    # Base Model with augmentation, Parameters can be changed here!
    network = create_model(3, 0.25, 'same', 2, True)
    train_or_load_network(network, training_images, training_labels)

#Train and evaluates model
if __name__ == '__main__':
    run()