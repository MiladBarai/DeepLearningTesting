from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras import models
from test.trainAndEvaluate import get_one_img, show_changed_images
from matplotlib import pyplot as plt

# File path to the first model trained in fold, of the base models
model_file_path = Path( "../resources/Deep Neural Network Training/NN[FilterSize=3,DropOut=0.25,Padding=same,DataAugmentation=False,maxPooling=True]/1.h5")
# Load model
network = models.load_model(model_file_path)
# print summary of what the model looks like
network.summary()

found_conv = False
last_conv_layer_indexies = []

# Run through all layers
for i in range(len(network.layers)):
    # If not a conv layer skip it
    if 'conv' not in network.layers[i].name:
        # If we had a previous conv layer it was the last in the hidden layer stack of conv layers
        if found_conv:
            found_conv = False
            last_conv_layer_indexies.append(i - 1)
        continue
    # We found a conv layer
    else:
        found_conv = True

print("Conv blocks at layer indexies: "+str(last_conv_layer_indexies))
# redefine model to output after each last conv layer stack
network_outputs = [network.layers[i+1].output for i in last_conv_layer_indexies]

network = Model(inputs=network.inputs, outputs=network_outputs)

# Load img
img = get_one_img(1)
show_changed_images([img])

# Predict one img and get the output
feature_maps = network.predict(img)

column_size = 8
row_size = 16
layer_index = 1
# plot output from each network output
for feature_map in feature_maps:
    # plot all 128 maps
    img_index = 1
    print("Output from Convolution layer block: "+str(layer_index))
    # Largen plt figure
    plt.figure(figsize=(20, 20))

    for _ in range(column_size):

        for _ in range(row_size):
            # specify subplot
            feature_map_plot = plt.subplot(column_size, row_size, img_index)

            # remove the x and y axis text and incremental lines displaying x and y movements
            feature_map_plot.set_xticks([])
            feature_map_plot.set_yticks([])

            # plot feature map in grayscale
            plt.imshow(feature_map[0, :, :, img_index - 1], cmap='gray')

            img_index += 1

    layer_index += 1
    # show the figure
    plt.show()
