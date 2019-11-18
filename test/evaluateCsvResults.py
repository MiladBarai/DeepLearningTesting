import csv
import os
import math

root_directory = '../resources/Deep Neural Network Training'


def get_highest_acc_from_csv(csv_path):
    with open(csv_path) as csv_file:
        # Attach the csv to a csv reader
        csv_reader = csv.reader(csv_file)
        # Skip header with column names
        next(csv_reader, None)

        # variable for highest accuracy
        highest_acc = 0
        # go through all lines to get highest accuracy for this file

        for row in csv_reader:
            if highest_acc < float(row[3]):
                highest_acc = float(row[3])

    csv_file.close()
    return highest_acc


def get_map_of_best_result():
    nerualnet_results = {}
    # Walk through all subdirectories and gather all files
    for subdirectory, directories, files in os.walk(root_directory,):
        # for each file in the walked directories
        for file in files:

            # if it's not a csv file then skip it
            if not file.endswith(".csv"):
                continue

            # Get the neural-net config from the folder
            folders = subdirectory.split("\\")
            neural_net_config = folders[len(folders) - 1]

            # Get the file path of the file
            file_path = subdirectory + "\\" + file

            # Get highest validation accuracy in file
            highest_acc = get_highest_acc_from_csv(file_path)

            # If we have already encountered it add it to the existing value map
            if neural_net_config in nerualnet_results:
                nerualnet_results[neural_net_config].append(highest_acc)
            else:
                #print("Adding to neural-net results: " + neural_net_config)
                nerualnet_results.setdefault(neural_net_config, [highest_acc])
    return nerualnet_results

result_map = get_map_of_best_result()
network_statistics = {}

print("nn, mean, std")
for key in result_map.keys():
    result_length = len(result_map[key])

    # calculate mean
    total = 0
    for val in result_map[key]:
        total = total + val
    mean = total / result_length

    # calculate standard deviation
    delta = 0
    for val in result_map[key]:
        delta = delta + math.pow((val - mean), 2)
    std = math.sqrt(delta / result_length)

    # store results in map
    network_statistics.setdefault(key, (mean, std))
    print(key+","+str(mean)+","+str(std))
