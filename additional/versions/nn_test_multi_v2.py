import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import sys
from time import sleep
from datetime import datetime


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def sigmoid(x):
    return (1.0/(1.0+np.exp(-x)))

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))

    # result = np.log(((np.exp(x)) / (np.sum(np.exp(x), axis=1, keepdims=True))))
    result = exps / (np.sum(exps, axis=1, keepdims=True))

    if (np.any(np.isnan(result))):
        # print(x)
        print("Error in Softmax")
        exit()
    return result

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    dx = np.zeros(x.shape)
    dx[x < 0] = 0
    dx[x >= 1] = 1
    return dx.astype(float)

def rate_decay(alpha_0, decay_rate, n):
    result = (alpha_0 / (1.0 + decay_rate * n))
    return result

def log_loss(y_train, y_hat):
    result = (-(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat)))
    return result

def cross_entropy_loss(y_train, y_hat, epsilon=1e-11):
    m = y_train.shape[0]
    y_hat_clip = np.clip(y_hat, epsilon, 1 - epsilon)
    result = ((-1.0/m) * np.sum(np.sum(y_train * np.log(y_hat_clip), axis=1), axis=0))
    # likelihood = -np.log(y_hat[:, np.int_(y_train)])
    # result = (1.0 / m) * (np.sum(np.sum(likelihood, axis=1, keepdims=True), axis=0, keepdims=True))
    # print(y_train, y_hat)

    if (np.any(np.isnan(result))):
        print("Error in Cross Entropy")
        exit()

    return result

def forward(input, weight, bias, activation="sigmoid"):
    z = input.dot(weight) + bias.T    # (m x l1)

    if activation == "sigmoid":
        return sigmoid(z)  # (m x l1)
    elif activation == "softmax":
        return softmax(z)  # (m x l1)
    elif activation == "tanh":
        return tanh(z)  # (m x l1)
    elif activation == "relu":
        return relu(z)  # (m x l1)


def backward(inp, out, w_out, dz_out, activation="sigmoid"):
    if (activation == "sigmoid") or (activation == "softmax"):
        dz = out - dz_out    # Using w_out for y_train
        # dz = dz_out.dot(w_out.T) * d_sigmoid(out)               # (m x l1)
    elif activation == "tanh":
        dz = dz_out.dot(w_out.T) * d_tanh(out)               # (m x l1)
    elif activation == "relu":
        dz = dz_out.dot(w_out.T) * d_relu(out)               # (m x l1)

    # dz = dz_out.dot(w_out.T) * d_relu(out)               # (m x l1)
    dW = (inp.T).dot(dz)                             # (n x m) * (m x l1) = (n x l1)
    db = (np.sum(dz, axis=0, keepdims=True)).T             # (l1 x 1)
    return [dz, dW, db]


def main():
    iris = datasets.load_iris()
    X = iris.data   #[iris.target<2,:]
    y = iris.target.reshape(-1,1)

    # print(X.shape, y.shape)

    # One Hot Encoding
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Total Samples and Features
    m = X_train.shape[0]
    n = X_train.shape[1]
    # classes = y_hot.shape[0]

    # Weights and Biases
    # l1 = 4
    # l2 = 4
    # out = 3
    #
    # np.random.seed(int(time.time()%10))
    #
    # W1 = np.random.randn(n, l1) / np.sqrt(n)
    # b1 = np.zeros(l1).reshape(-1,1)
    #
    # W2 = np.random.randn(l1, l2) / np.sqrt(n)
    # b2 = np.zeros(l2).reshape(-1,1)
    #
    # W3 = np.random.randn(l2, out) / np.sqrt(l2)
    # b3 = np.zeros(out).reshape(-1,1)

    hidden_layers = [n,50,10,10,3]
    act = ['relu', 'relu', 'relu', 'softmax']

    no_of_layers = len(hidden_layers)-1
    dict = {}

    for i in range(1, len(hidden_layers)):
        dict['W' + str(i)] = np.random.randn(hidden_layers[i-1], hidden_layers[i]) / np.sqrt(n)
        dict['b' + str(i)] = np.zeros(hidden_layers[i]).reshape(-1,1)

    # print(dict)

    alpha_0 = 0.01
    decay_rate = 0.001
    iterations = 10000

    cost = []
    alphalist = []
    print(no_of_layers)

    for iter in range(iterations):

        alpha = rate_decay(alpha_0, decay_rate, i)
        alphalist.append(alpha)

        a = []
        a.append(X_train)
        # print(a[0].shape)

        for i in range(no_of_layers):
            a.append(forward(a[i], dict['W' + str(i+1)], dict['b' + str(i+1)], activation=act[i]))
            # print(a[i+1].shape)

        # exit()
        dz = []
        dW = []
        db = []
        dz.append(y_train)

        for i in range(no_of_layers,0,-1):
            if i==no_of_layers:
                [dzz, dWW, dbb] = backward(a[i-1], a[i], [], dz[no_of_layers-i], activation=act[i-1])
            else:
                [dzz, dWW, dbb] = backward(a[i-1], a[i], dict['W' + str(i+1)], dz[no_of_layers-i], activation=act[i-1])

            dz.append(dzz)
            dW.append(dWW)
            db.append(dbb)

        # Update Weights and Biases
        for i in range(1,no_of_layers+1):
            dict['W' + str(i)] -= (alpha * dW[no_of_layers-i])
            dict['b' + str(i)] -= (alpha * db[no_of_layers-i])

        # Calculate loss and cost
        loss = cross_entropy_loss(y_train, a[-1])
        cost.append(loss)

        if ((iter%(iterations/10)) == 0):
            print("Epoch", int(iter/(iterations/10)), ": ", loss)
        # printProgressBar(i + 1, iterations, prefix = 'Training:', suffix = 'Complete', length = 50)

    a1 = forward(X_test, dict['W' + str(1)], dict['b' + str(1)], activation=act[0])
    a2 = forward(a1, dict['W' + str(2)], dict['b' + str(2)], activation=act[1])
    a3 = forward(a2, dict['W' + str(3)], dict['b' + str(3)], activation=act[2])
    y_hat = forward(a3, dict['W' + str(4)], dict['b' + str(4)], activation=act[3])

    loss = cross_entropy_loss(y_test, y_hat)
    J = (1 / m) * np.sum(loss)
    print("\nTraining Loss: ", cost[-1])
    print("\nValidation Loss: ", loss)
    print("Sample Prediction: \n", y_test[10:20], "\n", y_hat[10:20])

    plt.plot(cost)
    plt.show()

if __name__== "__main__":
    main()
