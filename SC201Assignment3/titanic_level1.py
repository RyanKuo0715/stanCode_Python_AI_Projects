"""
File: titanic_level1.py
Name: Ryan Kuo
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle website. This model is the most flexible among all
levels. You should do hyper-parameter tuning to find the best model.
"""

import math
from collections import defaultdict
from util import *

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
    """
    :param filename: str, the filename to be processed
    :param data: an empty Python dictionary
    :param mode: str, indicating if it is training mode or testing mode
    :param training_data: dict[str: list], key is the column name, value is its data
                          (You will only use this when mode == 'Test')
    :return data: dict[str: list], key is the column name, value is its data
    """
    with open(filename, 'r') as f:
        first = True
        for line in f:
            if first:
                key_extractor(data, line.strip(), mode)
                first = False
            else:
                feature_extractor(data, line.strip(), mode, training_data)
    return data


def key_extractor(data, line, mode):
    pos = [1, 2, 4, 5, 6, 7, 9, 11]
    data_lst = line.split(',')
    start = shift = 0 if mode == 'Train' else 1
    for i in range(start, len(pos)):
        data[data_lst[pos[i]-shift]] = []


def feature_extractor(data, line, mode, training_data):
    pos = [(1, 'Survived'), (2, 'Pclass'), (5, 'Sex'), (6, 'Age'), (7, 'SibSp'),
           (8, 'Parch'), (10, 'Fare'), (12, 'Embarked')]
    data_lst = line.split(',')
    if_nan = False
    start = shift = 0 if mode == 'Train' else 1

    if mode == 'Train':
        for i in range(start, len(pos)):
            if not data_lst[pos[i][0]]:
                if_nan = True
                break

    if not if_nan:
        for i in range(start, len(pos)):
            if pos[i][0] == 1:
                data[pos[i][1]].append(int(data_lst[pos[i][0]]))
            elif pos[i][0] == 2:
                data[pos[i][1]].append(int(data_lst[pos[i][0]-shift]))
            elif pos[i][0] == 5:
                if data_lst[pos[i][0]-shift] == 'male':
                    data[pos[i][1]].append(1)
                elif data_lst[pos[i][0]-shift] == 'female':
                    data[pos[i][1]].append(0)
            elif pos[i][0] == 6:
                if data_lst[pos[i][0]-shift]:
                    data[pos[i][1]].append(float(data_lst[pos[i][0]-shift]))
                else:
                    total = sum(training_data[pos[i][1]][j] for j in range(len(training_data[pos[i][1]])))
                    data[pos[i][1]].append(round(total/len(training_data[pos[i][1]]), 3))
            elif pos[i][0] == 7:
                data[pos[i][1]].append(int(data_lst[pos[i][0]-shift]))
            elif pos[i][0] == 8:
                data[pos[i][1]].append(int(data_lst[pos[i][0]-shift]))
            elif pos[i][0] == 10:
                if data_lst[pos[i][0]-shift]:
                    data[pos[i][1]].append(float(data_lst[pos[i][0]-shift]))
                else:
                    total = sum(training_data[pos[i][1]][j] for j in range(len(training_data[pos[i][1]])))
                    data[pos[i][1]].append(round(total/len(training_data[pos[i][1]]), 3))
            elif pos[i][0] == 12:
                if data_lst[pos[i][0]-shift] == 'S':
                    data[pos[i][1]].append(0)
                elif data_lst[pos[i][0]-shift] == 'C':
                    data[pos[i][1]].append(1)
                elif data_lst[pos[i][0]-shift] == 'Q':
                    data[pos[i][1]].append(2)


def one_hot_encoding(data: dict, feature: str):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: dict[str, list], remove the feature column and add its one-hot encoding features
    """
    if feature == 'Sex':
        data['Sex_0'] = []
        data['Sex_1'] = []
        for value in data[feature]:
            if value == 0:
                data['Sex_0'].append(1)
                data['Sex_1'].append(0)
            elif value == 1:
                data['Sex_0'].append(0)
                data['Sex_1'].append(1)
        data.pop('Sex')
    elif feature == 'Pclass':
        data['Pclass_0'] = []
        data['Pclass_1'] = []
        data['Pclass_2'] = []
        for value in data[feature]:
            if value == 1:
                data['Pclass_0'].append(1)
                data['Pclass_1'].append(0)
                data['Pclass_2'].append(0)
            elif value == 2:
                data['Pclass_0'].append(0)
                data['Pclass_1'].append(1)
                data['Pclass_2'].append(0)
            elif value == 3:
                data['Pclass_0'].append(0)
                data['Pclass_1'].append(0)
                data['Pclass_2'].append(1)
        data.pop('Pclass')
    elif feature == 'Embarked':
        data['Embarked_0'] = []
        data['Embarked_1'] = []
        data['Embarked_2'] = []
        for value in data[feature]:
            if value == 0:
                data['Embarked_0'].append(1)
                data['Embarked_1'].append(0)
                data['Embarked_2'].append(0)
            elif value == 1:
                data['Embarked_0'].append(0)
                data['Embarked_1'].append(1)
                data['Embarked_2'].append(0)
            elif value == 2:
                data['Embarked_0'].append(0)
                data['Embarked_1'].append(0)
                data['Embarked_2'].append(1)
        data.pop('Embarked')
    return data


def normalize(data: dict):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :return data: dict[str, list], key is the column name, value is its normalized data
    """
    pos = ['Age', 'SibSp', 'Parch', 'Fare']
    for attribute in pos:
        data[attribute] = list((element-min(data[attribute]))/(max(data[attribute])-min(data[attribute]))
                               for element in data[attribute])
    return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param degree: int, degree of polynomial features
    :param num_epochs: int, the number of epochs for training
    :param alpha: float, known as step size or learning rate
    :return weights: dict[str, float], feature name and its weight
    """
    # Step 1 : Initialize weights
    weights = {}  # feature => weight
    keys = list(inputs.keys())
    if degree == 1:
        for i in range(len(keys)):
            weights[keys[i]] = 0
    elif degree == 2:
        for i in range(len(keys)):
            weights[keys[i]] = 0
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                weights[keys[i] + keys[j]] = 0

    # Step 2 : Start training
    for epoch in range(num_epochs):
        for i in range(len(labels)):
            temp = defaultdict(int)
            # Step 3 : Feature Extract
            if degree == 1 or degree == 2:
                for j in range(len(keys)):
                    temp[keys[j]] = inputs[keys[j]][i]
                    if degree == 2:
                        for k in range(j, len(keys)):
                            temp[keys[j]+keys[k]] = inputs[keys[j]][i]*inputs[keys[k]][i]
            # Step 4 : Update weights
            y = labels[i]
            k = dotProduct(weights, temp)
            h = 1/(1+math.exp(-k))
            increment(weights, -alpha*(h-y), temp)

    return weights
