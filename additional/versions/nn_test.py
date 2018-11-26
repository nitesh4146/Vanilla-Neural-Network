import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return (1.0/(1.0+np.exp(-x)))

def main():
    iris = datasets.load_iris()
    X = iris.data[iris.target<2,:]
    y = iris.target[iris.target<2].reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # print(X.shape, y.shape)

    # One Hot Encoding
    # enc = OneHotEncoder()
    # enc.fit(y)
    # y_hot = enc.transform(y).toarray()

    # Total Samples and Features
    m = X_train.shape[0]
    n = X_train.shape[1]
    # classes = y_hot.shape[0]

    # Weights and Biases
    W = np.random.random(n).reshape(-1,1)
    b = np.random.random(1)
    alpha = 0.1
    dz = np.zeros(y_train.shape)
    dW = np.zeros(W.shape)
    cost = []

    for i in range(100):
        # Forward Propagation
        z = np.matmul(X_train, W) + b
        y_hat = sigmoid(z)
        loss = - (y_train * np.log(y_hat) + (1 - y_train) * np.log(1-y_hat))
        J = (1 / m) * np.sum(loss)
        cost.append(J)

        # Backward Propagation
        dz = y_hat - y_train
        dW = np.matmul(np.transpose(X_train), dz)/m
        db = np.sum(dz)/m
        W = W - (alpha * dW)
        b = b - (alpha * db)

    z = np.matmul(X_test, W) + b
    y_hat = sigmoid(z)
    loss = - (y_test * np.log(y_hat) + (1 - y_test) * np.log(1-y_hat))
    J = (1 / m) * np.sum(loss)
    print(J)
    plt.plot(cost)
    plt.show()
    # print(y_hat)

if __name__== "__main__":
    main()
