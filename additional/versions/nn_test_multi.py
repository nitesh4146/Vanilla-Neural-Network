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
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
    result = ((-1.0/m) * np.sum(np.sum(y_train * np.log(y_hat), axis=1), axis=0))
    # likelihood = -np.log(y_hat[:, np.int_(y_train)])
    # result = (1.0 / m) * (np.sum(np.sum(likelihood, axis=1, keepdims=True), axis=0, keepdims=True))
    # print(y_train, y_hat)

    if (np.any(np.isnan(result))):
        print("Error in Cross Entropy")
        exit()

    return result

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
    l1 = 100
    out = 3

    np.random.seed(0)
    # W1 = np.random.random((n, l1))
    W1 = np.random.randn(n, l1) / np.sqrt(n)
    # W_out = np.random.random((l1, out))
    W_out = np.random.randn(l1, out) / np.sqrt(l1)

    b1 = np.zeros(l1).reshape(-1,1)
    b_out = np.zeros(out).reshape(-1,1)

    alpha_0 = 0.1
    decay_rate = 0.001
    iterations = 100

    cost = []
    alphalist = []

    for i in range(iterations):

        alpha = rate_decay(alpha_0, decay_rate, i)
        alphalist.append(alpha)

        # Forward Propagation Layer 1
        z1 = X_train.dot(W1) + b1.T    # (m x l1)
        a1 = relu(z1)  # (m x l1)
        # a1 = np.tanh(z1)  # (m x l1)

        # Forward Propagation Layer 2
        z_out = a1.dot(W_out) + b_out.T    # (m x out)
        y_hat = softmax(z_out)

        # Calculate loss and cost
        loss = cross_entropy_loss(y_train, y_hat)
        # J = (1 / m) * np.sum(loss)
        # print(loss)
        cost.append(loss)

        # Backward Propagation
        dz_out = y_hat - y_train                                                # (m x out)
        # dz_out = y_hat
        # dz_out[:, np.int_(y_train)] -= 1
        dW_out = (a1.T).dot(dz_out)                        # (l1 x m) * (m x out) = (l1 x out)
        db_out = (np.sum(dz_out, axis=0, keepdims=True)).T        # (out x 1)

        dz1 = dz_out.dot(W_out.T) * d_relu(a1)               # (m x l1)
        # dz1 = dz_out.dot(np.transpose(W_out)) * (1 - np.power(a1, 2))               # (m x l1)
        dW1 = (X_train.T).dot(dz1)                             # (n x m) * (m x l1) = (n x l1)
        db1 = (np.sum(dz1, axis=0, keepdims=True)).T             # (l1 x 1)

        # Update Weights and Biases
        W1 = W1 - (alpha * dW1)
        b1 = b1 - (alpha * db1)
        W_out = W_out - (alpha * dW_out)
        b_out = b_out - (alpha * db_out)

        printProgressBar(i + 1, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)

    z1 = X_test.dot(W1) + np.transpose(b1)
    a1 = relu(z1)
    # a1 = np.tanh(z1)
    z_out = a1.dot(W_out) + np.transpose(b_out)
    y_hat = softmax(z_out)

    loss = cross_entropy_loss(y_test, y_hat)
    J = (1 / m) * np.sum(loss)
    print("\nTraining Loss: ", cost[-1])
    print("\nValidation Loss: ", loss)
    print("Sample Prediction: \n", y_test[5:10], "\n", y_hat[5:10])

    plt.plot(cost)
    plt.show()
    # print(d_relu(np.array([[-2, 3, 4], [1, -5, 2]])))

    sys.stdout.write("\n")
if __name__== "__main__":
    main()
