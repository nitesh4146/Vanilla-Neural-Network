import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix
import time
import sys
from time import sleep
from datetime import datetime


def printProgressBar (cost, decrease, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    # percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if decrease:
        arrow = u'\u2191'
    else:
        arrow = u'\u2193'

    print('\r%s |%s| %s %.4f (%s)' % (prefix, bar, "Cost: ", cost, arrow), end = '\r')

    # Print New Line on Complete
    if iteration == total:
        print()


def draw_network(layers, act):
    print("\x1b[1;36;40m")
    print("\n\t***** Your Network *****")
    print("input" + "(" + str(layers[0]) + ") ==>", end=" ")

    for l in range(1, len(layers)-1):
        print("L" + str(l) + "(" + str(layers[l]) + ") ==>", end=" ")

    print("out", "(" + str(layers[-1]) + ")")
    print("\n\x1b[0m")

def sigmoid(x):
    # x = np.clip(x, epsilon, 1 - epsilon)
    return (1.0/(1.0+np.exp(-x)))

def d_sigmoid(x):
    a = sigmoid(x)
    return a * (1 - a)

def d_tanh(x):
    a = 1 - np.power(np.tanh(x), 2)
    return a

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))

    # result = np.log(((np.exp(x)) / (np.sum(np.exp(x), axis=1, keepdims=True))))
    result = exps / (np.sum(exps, axis=1, keepdims=True))

    if (np.any(np.isnan(result))):
        print("Error in Softmax")
        exit()
    return result

def relu(x):
    return np.maximum(0.0, x)

def d_relu(x):
    dx = np.zeros(x.shape)
    dx[x < 0] = 0.0
    dx[x >= 1] = 1.0
    return dx.astype(float)

def rate_decay(alpha_0, decay_rate, n):
    result = (alpha_0 / (1.0 + decay_rate * n))
    return result

def log_loss(y_train, y_hat):
    result = (-(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat)))
    return result

def cross_entropy_loss(y_train, y_hat, epsilon=1e-11):
    m = y_train.shape[0]
    n = y_train.shape[1]

    y_hat_clip = np.clip(y_hat, epsilon, 1 - epsilon)
    result = ((-1.0/ (m)) * np.sum(np.sum(y_train * np.log(y_hat_clip), axis=1), axis=0))

    if (np.any(np.isnan(result))):
        print("Error in Cross Entropy")
        exit()

    return result

def logistic_loss(y_train, y_hat, epsilon=1e-11):
    m = y_train.shape[0]
    n = y_train.shape[1]
    y_hat_clip = y_hat
    # y_hat_clip = np.clip(y_hat, epsilon, 1 - epsilon)

    loss = - (y_train * np.log(y_hat_clip) + (1 - y_train) * np.log(1-y_hat_clip))
    result = (1.0 / m) * np.sum(loss)

    if (np.any(np.isnan(result))):
        print("Error in logistic_loss")
        exit()
    return result


def l1_reg(x, lam):
    return (lam * np.abs(x))

def l2_reg(x, lam):
    # print((lam * np.sum(np.power(x, 2))) / 2.0)
    return (lam * np.power(x, 2)) / 2.0

def forward(input, weight, bias, activation="sigmoid"):
    z = input.dot(weight) + bias.T    # (m x l1)

    if activation == "sigmoid":
        return sigmoid(z)  # (m x l1)
    elif activation == "softmax":
        return softmax(z)  # (m x l1)
    elif activation == "tanh":
        return np.tanh(z)  # (m x l1)
    elif activation == "relu":
        return relu(z)  # (m x l1)


def backward(inp, out, w_out, dz_out, activation="sigmoid", output_layer=False):
    m = inp.shape[0]

    #   Only for sigmoid or softmax in output layer
    if output_layer:
        dz = out - dz_out                                       # Using w_out for y_train
    else:
        if (activation == "sigmoid"):
            dz = dz_out.dot(w_out.T) * d_sigmoid(out)           # (m x l1)
        elif activation == "tanh":
            dz = dz_out.dot(w_out.T) * d_tanh(out)              # (m x l1)
        elif activation == "relu":
            dz = dz_out.dot(w_out.T) * d_relu(out)              # (m x l1)

    dW = (1.0 / m) * (inp.T).dot(dz)                            # (n x m) * (m x l1) = (n x l1)
    db = (1.0 / m) * (np.sum(dz, axis=0, keepdims=True)).T      # (l1 x 1)
    return [dz, dW, db]


def predict(X_test, dict, act, no_of_layers):
    a = []
    a.append(X_test)

    for i in range(no_of_layers):
        a.append(forward(a[i], dict['W' + str(i+1)], dict['b' + str(i+1)], activation=act[i]))
    return a[-1]

def normalize(x):
    mean = np.mean(x, axis=0)
    deviation = np.amax(x, axis=0) - np.amin(x, axis=0)
    return (x - mean)/deviation

def confusion_matrix(y, y_pred):
    cm = np.zeros(shape=(2,2))
    for a, p in zip(y, y_pred):
        cm[int(a),int(p)] += 1
    return cm.ravel()

def scores(y, y_pred):
    y_pred[y_pred == np.amax(y_pred, axis=1).reshape(-1,1)] = 1
    y_pred[y_pred < np.amax(y_pred, axis=1).reshape(-1,1)] = 0

    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for i in range(y.shape[1]):
        tn_class, fp_class, fn_class, tp_class = confusion_matrix(y[:,i], y_pred[:,i])
        tp += tp_class
        tn += tn_class
        fp += fp_class
        fn += fn_class

    # Sensitivity, hit rate, recall, or true positive rate
    sensitivity = tp/(tp+fn)
    # Specificity or true negative rate
    specificity = tn/(tn+fp)
    # Precision or positive predictive value
    precision = tp/(tp+fp)
    # Negative predictive value
    npv = tn/(tn+fn)
    # Fall out or false positive rate
    fpr = fp/(fp+tn)
    # False negative rate
    fnr = fn/(tp+fn)
    # False discovery rate
    fdr = fp/(tp+fp)

    # Overall accuracy
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    return accuracy

def main():
    try:
        iris = datasets.load_iris()
        X = iris.data.astype(float)   #[iris.target<2,:]
        y = iris.target.reshape(-1,1)
        X = normalize(X)

        # One Hot Encoding
        enc = OneHotEncoder()
        enc.fit(y)
        y = enc.transform(y).toarray().astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Total Samples and Features
        samples = X_train.shape[0]
        features = X_train.shape[1]
        classes = y.shape[1]

        hidden_layers = [features, 100, 60, classes]
        # act = ['relu', 'relu', 'softmax']
        # act = ['tanh', 'tanh', 'tanh', 'softmax']
        act = ['sigmoid', 'sigmoid', 'softmax']

        draw_network(hidden_layers, act)
        prev_cost = -999

        no_of_layers = len(hidden_layers) - 1
        dict = {}

        for i in range(1, len(hidden_layers)):
            eps = math.sqrt(6) / (math.sqrt(hidden_layers[i-1] + hidden_layers[i]))
            dict['W' + str(i)] = 2 * eps * np.random.uniform(hidden_layers[i], (1 + hidden_layers[i-1]) , size=(hidden_layers[i-1], hidden_layers[i])) - eps

            # dict['W' + str(i)] = np.random.randn(hidden_layers[i-1], hidden_layers[i]) #/ np.sqrt(classes)
            dict['b' + str(i)] = np.zeros(hidden_layers[i]).reshape(-1,1)

        alpha_0 = 0.01
        decay_rate = 0.001
        iterations = 10000
        lamb_da = 0.01
        loss_fn = "entropy"
        cost = []
        alphalist = []

        # Training starts here

        for iter in range(iterations):

            alpha = rate_decay(alpha_0, decay_rate, i)
            alphalist.append(alpha)

            a = []
            a.append(X_train)

            # Forward Pass
            for i in range(no_of_layers):
                a.append(forward(a[i], dict['W' + str(i+1)], dict['b' + str(i+1)], activation=act[i]))

            dz = []
            dW = []
            db = []
            dz.append(y_train)

            # Backpropagation of Gradients
            for i in range(no_of_layers,0,-1):
                if i==no_of_layers:
                    [dzz, dWW, dbb] = backward(a[i-1], a[i], [], dz[no_of_layers-i], activation=act[i-1], output_layer=True)
                else:
                    [dzz, dWW, dbb] = backward(a[i-1], a[i], dict['W' + str(i+1)], dz[no_of_layers-i], activation=act[i-1])

                dz.append(dzz)
                dW.append(dWW)
                db.append(dbb)

            # Update Weights and Biases
            for i in range(1, no_of_layers + 1):
                # dW[no_of_layers-i] += l1_reg(dict['W' + str(i)], lamb_da)
                # dW[no_of_layers-i] += l2_reg(dict['W' + str(i)], lamb_da)
                dict['W' + str(i)] -= (alpha * dW[no_of_layers-i])
                dict['b' + str(i)] -= (alpha * db[no_of_layers-i])

            # Calculate loss and cost
            if ((iter%(iterations/10)) == 0):
                if loss_fn == "entropy":
                    loss = cross_entropy_loss(y_train, a[-1])
                elif loss_fn == "logistic":
                    loss = logistic_loss(y_train, a[-1])

                if loss <= prev_cost:
                    decrease = True
                else:
                    decrease = False

                cost.append(loss)
                prev_cost = cost[-1]
                # print("Epoch", int(iter/(iterations/10)+1), ": ", loss)

            printProgressBar(loss, decrease, iter+1, iterations, prefix = 'Training:', suffix = 'Complete', length = 50)

        # Training ends here

        y_hat = predict(X_train, dict, act, no_of_layers)
        if loss_fn == "entropy":
            loss = cross_entropy_loss(y_train, y_hat)
        elif loss_fn == "logistic":
            loss = logistic_loss(y_train, y_hat)

        print("Train Accuracy: %.3f%%"%(scores(y_train, y_hat)*100))

        y_hat = predict(X_test, dict, act, no_of_layers)
        if loss_fn == "entropy":
            loss = cross_entropy_loss(y_test, y_hat)
        elif loss_fn == "logistic":
            loss = logistic_loss(y_test, y_hat)
        print("Test Accuracy: %.3f%%"%(scores(y_test, y_hat)*100))

        # loss = logistic_loss(y_test, y_hat)
        # print("\nTraining Loss: ", cost[-1])
        # print("\nValidation Loss: ", loss)
        # print("Sample Prediction: \n", y_test[10:20], "\n", y_hat[10:20])

        # plt.plot(cost)
        # plt.show()

    except (KeyboardInterrupt, SystemExit):
        print("\n\x1b[1;31;40m \t Execution Stopped by User")
        print("\n\x1b[0m")
    # except:
    #     print("\n\x1b[1;31;40m \t Unexpected Error Occurred!")
    #     print("\n\x1b[0m")

if __name__== "__main__":
    main()
