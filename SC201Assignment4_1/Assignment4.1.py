"""
File: titanic_deep_nn.py
Name: Ryan Kuo
-----------------------------
This file demonstrates how to create a deep
neural network (5 layers NN) to train our
titanic data. Your code should use all the
constants and global variables.
You should see the following Acc if you
correctly implement the deep neural network
Acc: 0.8431372549019608
-----------------------------
X.shape = (N0, m)
Y.shape = (1, m)
W1.shape -> (N0, N1)
W2.shape -> (N1, N2)
W3.shape -> (N2, N3)
W4.shape -> (N3, N4)
W5.shape -> (N4, N5)
B1.shape -> (N1, 1)
B2.shape -> (N2, 1)
B3.shape -> (N3, 1)
B4.shape -> (N4, 1)
B5.shape -> (N5, 1)
"""

from collections import defaultdict
import numpy as np

# Constants
TRAIN = 'titanic_data/train.csv'     # This is the filename of interest
NUM_EPOCHS = 40000                   # This constant controls the total number of epochs
ALPHA = 0.01                         # This constant controls the learning rate α
L = 5                                # This number controls the number of layers in NN
NODES = {                            # This Dict[str: int] controls the number of nodes in each layer
    'N0': 6,
    'N1': 5,
    'N2': 4,
    'N3': 3,
    'N4': 2,
    'N5': 1,
}


def main():
    """
    Print out the final accuracy of your deep neural network!
    You should see 0.8431372549019608
    """
    X_train, Y = data_preprocessing()
    _, m = X_train.shape
    print('Y.shape', Y.shape)
    print('X.shape', X_train.shape)
    # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    X = normalize(X_train)
    W, B = neural_network(X, Y)
    # Training Acc
    K_train = {}
    A_train = {'A0': X}
    for i in range(1, L):
        K_train[f'K{i}'] = np.dot(W[f'W{i}'].T, A_train[f'A{i-1}']) + B[f'B{i}']
        A_train[f'A{i}'] = np.maximum(0, K_train[f'K{i}'])
    K_train[f'A{L}'] = np.dot(W[f'W{L}'].T, A_train[f'A{L-1}']) + B[f'B{L}']
    scores = np.dot(W[f'W{L}'].T, A_train[f'A{L-1}']) + B[f'B{L}']
    predictions = np.where(scores > 0, 1, 0)
    acc = np.equal(predictions, Y)
    num_acc = np.sum(acc)
    print('Training Acc:', num_acc/m)


def normalize(X):
    """
    :param X: numpy_array, the dimension is (num_phi, m)
    :return: numpy_array, the values are normalized, where the dimension is still (num_phi, m)
    """
    min_array = np.min(X, axis=1, keepdims=True)
    max_array = np.max(X, axis=1, keepdims=True)
    return (X - min_array) / (max_array - min_array)


def neural_network(X, Y):
    """
    :param X: numpy_array, the array holding all the training data
    :param Y: numpy_array, the array holding all the ture labels in X
    :return (weights, bias): the tuple of parameters of this deep NN
             weights: Dict[str, float], key is 'W1', 'W2', ...
                                        value is the corresponding float
             bias: Dict[str, float], key is 'B1', 'B2', ...
                                     value is the corresponding float
    """
    n, m = X.shape
    np.random.seed(1)
    weights = {}
    biases = {}

    # Initialize all the weights and biases
    for i in range(1, L+1):
        weights[f'W{i}'] = np.random.rand(NODES[f'N{i-1}'], NODES[f'N{i}']) - 0.5
        biases[f'B{i}'] = np.random.rand(NODES[f'N{i}'], 1) - 0.5

    K = {}
    A = {}
    dA = {}
    dK = {}
    dW = {}
    dB = {}
    print_every = 1000
    for epoch in range(NUM_EPOCHS):
        # Forward Pass
        A['A0'] = X
        for i in range(1, L):
            K[f'K{i}'] = np.dot(weights[f'W{i}'].T, A[f'A{i-1}']) + biases[f'B{i}']
            A[f'A{i}'] = np.maximum(0, K[f'K{i}'])
        scores = np.dot(weights[f'W{L}'].T, A[f'A{L-1}']) + biases[f'B{L}']
        H = 1/(1+np.exp(-scores))
        J = (1/m)*np.sum(-(Y*np.log(H)+(1-Y)*np.log(1-H)))
        if epoch % print_every == 0:
            print('Cost:', J)

        # Backward Pass
        for i in range(L, 0, -1):
            if i != L:
                dA[f'dA{i}'] = np.dot(weights[f'W{i+1}'], dK[f'dK{i+1}'])
            if i == L:
                dK[f'dK{i}'] = (1/m)*np.sum(H-Y, axis=0, keepdims=True)
            else:
                dK[f'dK{i}'] = dA[f'dA{i}']*np.where(K[f'K{i}'] > 0, 1, 0)
            dW[f'dW{i}'] = np.dot(A[f'A{i-1}'], dK[f'dK{i}'].T)
            dB[f'dB{i}'] = np.sum(dK[f'dK{i}'], axis=1, keepdims=True)

        # Updates all the weights and biases
        for i in range(1, L+1):
            weights[f'W{i}'] -= ALPHA*dW[f'dW{i}']
            biases[f'B{i}'] -= ALPHA*dB[f'dB{i}']

    return weights, biases


def data_preprocessing(mode='train'):
    """
    :param mode: str, indicating if it's training mode or testing mode
    :return: Tuple(numpy_array, numpy_array), the first one is X, the other one is Y
    """
    data_lst = []
    label_lst = []
    first_data = True
    if mode == 'train':
         with open(TRAIN, 'r') as f:
            for line in f:
                data = line.split(',')
                # ['0PassengerId', '1Survived', '2Pclass', '3Last Name', '4First Name', '5Sex', '6Age', '7SibSp', '8Parch', '9Ticket', '10Fare', '11Cabin', '12Embarked']
                if first_data:
                    first_data = False
                    continue
                if not data[6]:
                    continue
                label = [int(data[1])]
                if data[5] == 'male':
                    sex = 1
                else:
                    sex = 0
                # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
                passenger_lst = [int(data[2]), sex, float(data[6]), int(data[7]), int(data[8]), float(data[10])]
                data_lst.append(passenger_lst)
                label_lst.append(label)
    else:
        pass
    return np.array(data_lst).T, np.array(label_lst).T


if __name__ == '__main__':
    main()
