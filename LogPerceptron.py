#!/usr/bin/python


# Implementation of multi-class logistic perceptron algorithm using softmax function, from scratch
import random
import math
import copy


# Load file and read data
def get_dataset(filename):
    dataset = list()
    with open(filename) as f:
        dataset = f.readlines()
    dataset = [d.split(',') for d in dataset]
    for row in dataset:
        if row[-1] == "Slight-Right-Turn\n":
            row[-1] = [1,0,0,0]
        elif row[-1] == "Sharp-Right-Turn\n":
            row[-1] = [0,1,0,0]
        elif row[-1] == "Slight-Left-Turn\n":
            row[-1] = [0,0,1,0]
        else:
            row[-1] = [0,0,0,1]
        for col in range(len(row)-1):
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


# Generate output class result
def classify(prediction_row):
    m = max(prediction_row)
    output_class = [0,0,0,0]
    for i in range(len(prediction_row)):
        if prediction_row[i] == m:
            output_class[i] = 1
            break
    return output_class


# Update weights based on gradient of likelihood
def update_weights(weights, row, prediction_row, l_rate):
    actual_result = row[-1]
    for i in range(len(actual_result)):
        if actual_result[i] <= prediction_row[i]:
            sign = 1
        else:
            sign = -1
        weights[i][0] = weights[i][0] + sign*l_rate/float(len(prediction_row))
        for j in range(len(row)-1):
            weights[i][j+1] = weights[i][j+1] + sign*l_rate*(1-prediction_row[i])*row[j]/float(len(prediction_row))
    return weights


# Predict output class likelihoods using softmax function
def predict(row, weights):
    likelihood = list()
    for i in range(len(weights)):
        score = weights[i][0]
        for j in range(len(weights[0])-1):
            score = score + weights[i][j+1]*row[j]
        likelihood.append(math.exp(-score))
    s = sum(likelihood)
    prediction_row = [l/s for l in likelihood]
    return prediction_row


# Train the weights of the perceptron
def train(training_set, l_rate, n_epoch):
    n_col = len(training_set[0])
    n_row = len(training_set[0][-1])
    weights = [[0.0 for i in range(n_col)] for j in range(n_row)]
    for epoch in range(n_epoch):
        for row in training_set:
            prediction_row = predict(row, weights)
            weights = update_weights(weights, row, prediction_row, l_rate)
    return weights


# Call logistic perceptron algorithm
def perceptron(training_set, test_set, l_rate, n_epoch):
    predictions = list()
    weights = train(training_set, l_rate, n_epoch)
    for row in test_set:
        prediction_row = predict(row, weights)
        output_class = classify(prediction_row)
        predictions.append(output_class)
    return predictions


# Calculate percentage accuracy of obtained results
def calculate_accuracy(actual_results, predictions):
    count_correct = 0
    for i in range(len(actual_results)):
        if actual_results[i] == predictions[i]:
            count_correct += 1
    return (count_correct*100.0)/float(len(actual_results))


# Run perceptron algorithm on cross validation sets and report accuracy of obtained results
def run_algorithm(data_sets, n, l_rate, n_epoch):
    accuracy = list()
    for i in range(n):
        copy_of_data_sets = copy.copy(data_sets)
        test_set = copy_of_data_sets.pop(i)
        training_set = list()
        for j in range(n-1):
            training_set = training_set + copy_of_data_sets.pop()
        predictions = perceptron(training_set, test_set, l_rate, n_epoch)
        actual_results = [row[-1] for row in test_set]
        accuracy_i = calculate_accuracy(actual_results, predictions)
        accuracy.append(accuracy_i)
        print("%2.3f" % accuracy_i)
    return sum(accuracy)/float(len(accuracy))


def main():
    n = 10
    l_rate = 0.01
    n_epoch = 1
    filename = 'sensor_readings_4.data'
    dataset = get_dataset(filename)
    data_sets = cross_validation_split(dataset, n)
    accuracy_perceptron = run_algorithm(data_sets, n, l_rate, n_epoch)
    print("Multi-class logistic perceptron performance accuracy is %2.1f per cent" % accuracy_perceptron)


main()