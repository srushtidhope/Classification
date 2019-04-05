#!/usr/bin/python


# Implementation of k-nearest neighbours algorithm from scratch
import random
import math
import operator
import copy


# Load file and read data
def get_dataset(filename):
    dataset = list()
    with open(filename) as f:
        dataset = f.readlines()
    dataset = [d.split(',') for d in dataset]
    for row in dataset:
        if row[-1] == "Slight-Right-Turn\n":
            row[-1] = "1"
        elif row[-1] == "Sharp-Right-Turn\n":
            row[-1] = "2"
        elif row[-1] == "Slight-Left-Turn\n":
            row[-1] = "3"
        else:
            row[-1] = "4"
        for col in range(len(row)):
            row[col] = float(row[col])
    return dataset


# Split dataset into n sets for cross validation
def cross_validation_split(dataset, n):
    set_size = int(len(dataset)/n)
    data_sets = list()
    for i in range(n-1):
        set_i = list()
        while len(set_i) < set_size:
            index = random.randrange(len(dataset))
            set_i.append(dataset.pop(index))
        data_sets.append(set_i)
    data_sets.append(dataset)
    return data_sets


# Calculate euclidean distance between two data points
def distance(row1, row2):
    dist = 0.0
    for i in range(len(row1)):
        dist += (row1[i] - row2[i]) ** 2
    return math.sqrt(dist)


# Get k nearest neighbours of a data point
def get_k_neighbours(training_set, row, k):
    dist = list()
    for row_i in training_set:
        d_i = distance(row_i, row)
        dist.append((row_i, d_i))
    dist.sort(key=operator.itemgetter(1))
    neighbours_i = list()
    for i in range(k):
        neighbours_i.append(dist[i][0])
    return neighbours_i


# Predict class to which the data point belongs
def predict(neighbours_i):
    class_count = {}
    for i in range(len(neighbours_i)):
        class_i = neighbours_i[i][-1]
        if class_i in class_count:
            class_count[class_i] += 1
        else:
            class_count[class_i] = 1
    class_sorted = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return class_sorted[0][0]


# k nearest neighbours algorithm - get neighbours and predict class
def knn(training_set, test_set, k):
    predictions = list()
    for row in test_set:
        neighbours_i = get_k_neighbours(training_set, row, k)
        prediction_i = predict(neighbours_i)
        predictions.append(prediction_i)
    return predictions


# Calculate percentage accuracy of obtained results
def calculate_accuracy(actual_results, predictions):
    count_correct = 0
    for i in range(len(actual_results)):
        if actual_results[i] == predictions[i]:
            count_correct += 1
    return (count_correct*100.0)/float(len(actual_results))


# Run k nearest neighbours algorithm on cross validation sets and report accuracy of obtained results
def run_algorithm(data_sets, n, k):
    accuracy = list()
    for i in range(n):
        copy_of_data_sets = copy.copy(data_sets)
        test_set = copy_of_data_sets.pop(i)
        training_set = list()
        for j in range(n-1):
            training_set = training_set + copy_of_data_sets.pop()
        predictions = knn(training_set, test_set, k)
        actual_results = [row[-1] for row in test_set]
        accuracy_i = calculate_accuracy(actual_results, predictions)
        accuracy.append(accuracy_i)
        print("%2.3f" % accuracy_i)
    return sum(accuracy)/float(len(accuracy))


def main():
    n = 10
    k = 50
    filename = 'sensor_readings_4.data'
    dataset = get_dataset(filename)
    data_sets = cross_validation_split(dataset, n)
    accuracy_knn = run_algorithm(data_sets, n, k)
    print("K-nearest neighbour performance accuracy is %2.1f per cent" % accuracy_knn)


main()