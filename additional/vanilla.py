import math
import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys


class vanilla:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.alpha_0 = 0.01
        self.loss_fn = "entropy"
        self.iterations = 0
        self.dict = {}
        self.decay_rate = 0.001


    def add_layer(self, units, input_dim=0, activation="sigmoid"):
        if input_dim == 0:
            self.layers.append(units)
            self.activations.append(activation)
        else:
            self.layers.append(input_dim)
            self.layers.append(units)
            self.activations.append(activation)


    def compile(self, learning_rate=0.01, decay_rate = 0.001, loss="entropy"):
        self.alpha_0 = learning_rate
        self.loss_fn = loss
        self.decay_rate = decay_rate


    def fit(self, X, y, validation_split=0.2, shuffle=True, nb_epoch=5, steps_per_epoch=100, lambda_ = 0.01, normalize=False, regularize=False):
        try:
            if normalize:
                X = self.normalize(X)

            self.validate(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split, random_state=42)

            # Total Samples and Features
            samples = X_train.shape[0]
            features = X_train.shape[1]
            classes = y.shape[1]

            self.iterations = nb_epoch * steps_per_epoch
            self.draw_network(self.layers, self.activations)

            no_of_layers = len(self.layers) - 1

            # Weight Initialization
            for i in range(1, len(self.layers)):
                # eps = math.sqrt(6) / (math.sqrt(self.layers[i-1] + self.layers[i]))
                # self.dict['W' + str(i)] = 2 * eps * np.random.uniform(self.layers[i], (1 + self.layers[i-1]) , size=(self.layers[i-1], self.layers[i])) - eps
                self.dict['W' + str(i)] = np.random.randn(self.layers[i-1], self.layers[i]) #/ np.sqrt(classes)
                self.dict['b' + str(i)] = np.zeros(self.layers[i]).reshape(-1,1)

            prev_cost = -999

            cost = []

            # Training starts here
            for iter in range(self.iterations):

                alpha = self.rate_decay(self.alpha_0, self.decay_rate, i)
                a = []
                a.append(X_train)

                # Forward Pass
                for i in range(no_of_layers):
                    a.append(self.forward(a[i], self.dict['W' + str(i+1)], self.dict['b' + str(i+1)], activation=self.activations[i]))

                dz = []
                dW = []
                db = []
                dz.append(y_train)

                # Backpropagation of Gradients
                for i in range(no_of_layers,0,-1):
                    if i==no_of_layers:
                        [dzz, dWW, dbb] = self.backward(a[i-1], a[i], [], dz[no_of_layers-i], activation=self.activations[i-1], output_layer=True)
                    else:
                        [dzz, dWW, dbb] = self.backward(a[i-1], a[i], self.dict['W' + str(i+1)], dz[no_of_layers-i], activation=self.activations[i-1])

                    dz.append(dzz)
                    dW.append(dWW)
                    db.append(dbb)

                # Update Weights and Biases
                for i in range(1, no_of_layers + 1):
                    if regularize:
                        dW[no_of_layers-i] += self.l1_reg(self.dict['W' + str(i)], lambda_)
                        # dW[no_of_layers-i] += self.l2_reg(self.dict['W' + str(i)], lambda_)
                    self.dict['W' + str(i)] -= (alpha * dW[no_of_layers-i])
                    self.dict['b' + str(i)] -= (alpha * db[no_of_layers-i])

                # Calculate loss and cost
                if ((iter%(self.iterations/nb_epoch)) == 0):
                    if self.loss_fn == "entropy":
                        loss = self.cross_entropy_loss(y_train, a[-1])
                    elif self.loss_fn == "logistic":
                        loss = self.logistic_loss(y_train, a[-1])

                    if loss <= prev_cost:
                        decrease = True
                    else:
                        decrease = False

                    cost.append(loss)
                    prev_cost = cost[-1]
                    # print("Epoch", int(iter/(iterations/10)+1), ": ", loss)

                self.printProgressBar(loss, decrease, iter+1, self.iterations, prefix = 'Training:', suffix = 'Complete', length = 50)

            y_pred_train = self.predict(X_train)
            if self.loss_fn == "entropy":
                loss = self.cross_entropy_loss(y_train, y_pred_train)
            elif self.loss_fn == "logistic":
                loss = self.logistic_loss(y_train, y_pred_train)

            print("\nTrain Accuracy: %.3f %%"%(self.scores(y_train, y_pred_train)['accuracy']*100))

            y_pred_test = self.predict(X_test)
            if self.loss_fn == "entropy":
                loss = self.cross_entropy_loss(y_test, y_pred_test)
            elif self.loss_fn == "logistic":
                loss = self.logistic_loss(y_test, y_pred_test)
            print("Validation Accuracy: %.3f %%"%(self.scores(y_test, y_pred_test)['accuracy']*100))

            return cost

        except (KeyboardInterrupt):
            print("\n\x1b[1;31;40mTraining Stopped by User")
            print("\n\x1b[0m")


    # Auxiliary Functions
    def printProgressBar (self, cost, decrease, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        if decrease:
            arrow = u'\u2191'
        else:
            arrow = u'\u2193'
        print('\r%s |%s| %s %.4f (%s)' % (prefix, bar, "Cost: ", cost, arrow), end = '\r')
        if iteration == total:
            print()


    def draw_network(self, layers, act):
        print("\x1b[1;36;40m")
        network = ''

        network += ("input" + "(" + str(layers[0]) + ") ==> ")
        for l in range(1, len(layers)-1):
            network += ("L" + str(l) + "(" + str(layers[l]) + ") ==> ")
        network += ("out(" + str(layers[-1]) + ")")
        nstar = int(len(network)/4) - 2
        print("\n" + nstar*" " + "**** Your Network ****")
        print(network)
        print("\x1b[0m")


    def validate(self, X, y):
        if (self.layers == []):
            print("\n\x1b[1;31;40m[Error] Please check your Layers")
            print("\x1b[0m")
            exit(0)
        elif (len(self.layers) != (len(self.activations) + 1)):
            print("\n\x1b[1;31;40m[Error] Input layer size undefined")
            print("\x1b[0m")
            exit(0)
        elif (X.shape[0] != y.shape[0]):
            print("\n\x1b[1;31;40m[Error] Data-Label size mismatch")
            print("\x1b[0m")
            exit(0)
        elif (y.shape[1] != self.layers[-1]):
            print("\n\x1b[1;31;40m[Error] Output layer mismatch")
            print("\x1b[0m")
            exit(0)

    def sigmoid(self, x):
        return (1.0/(1.0+np.exp(-x)))


    def d_sigmoid(self, x):
        a = self.sigmoid(x)
        return a * (1 - a)


    def d_tanh(self, x):
        a = 1 - np.power(np.tanh(x), 2)
        return a


    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        result = exps / (np.sum(exps, axis=1, keepdims=True))

        if (np.any(np.isnan(result))):
            print("Error in Softmax")
            exit()
        return result


    def relu(self, x):
        return np.maximum(0.0, x)


    def d_relu(self, x):
        dx = np.zeros(x.shape)
        dx[x < 0] = 0.0
        dx[x >= 1] = 1.0
        return dx.astype(float)


    def rate_decay(self, alpha_0, decay_rate, n):
        result = (alpha_0 / (1.0 + decay_rate * n))
        return result


    def cross_entropy_loss(self, y_train, y_hat, epsilon=1e-11):
        m = y_train.shape[0]
        n = y_train.shape[1]

        y_hat_clip = np.clip(y_hat, epsilon, 1 - epsilon)
        result = ((-1.0/ (m)) * np.sum(np.sum(y_train * np.log(y_hat_clip), axis=1), axis=0))

        if (np.any(np.isnan(result))):
            print("Error in Cross Entropy")
            exit()
        return result


    def logistic_loss(self, y_train, y_hat, epsilon=1e-11):
        m = y_train.shape[0]
        n = y_train.shape[1]

        loss = - (y_train * np.log(y_hat) + (1 - y_train) * np.log(1-y_hat))
        result = (1.0 / m) * np.sum(loss)

        if (np.any(np.isnan(result))):
            print("Error in logistic_loss")
            exit()
        return result


    def l1_reg(self, x, lam):
        return (lam * np.abs(x))


    def l2_reg(self, x, lam):
        return (lam * np.power(x, 2)) / 2.0


    def forward(self, input, weight, bias, activation="sigmoid"):
        z = input.dot(weight) + bias.T    # (m x l1)

        if activation == "sigmoid":
            return self.sigmoid(z)  # (m x l1)
        elif activation == "softmax":
            return self.softmax(z)  # (m x l1)
        elif activation == "tanh":
            return np.tanh(z)  # (m x l1)
        elif activation == "relu":
            return self.relu(z)  # (m x l1)


    def backward(self, inp, out, w_out, dz_out, activation="sigmoid", output_layer=False):
        m = inp.shape[0]

        #   Only for sigmoid or softmax in output layer
        if output_layer:
            dz = out - dz_out                                       # Using w_out for y_train
        else:
            if (activation == "sigmoid"):
                dz = dz_out.dot(w_out.T) * self.d_sigmoid(out)           # (m x l1)
            elif activation == "tanh":
                dz = dz_out.dot(w_out.T) * self.d_tanh(out)              # (m x l1)
            elif activation == "relu":
                dz = dz_out.dot(w_out.T) * self.d_relu(out)              # (m x l1)

        dW = (1.0 / m) * (inp.T).dot(dz)                            # (n x m) * (m x l1) = (n x l1)
        db = (1.0 / m) * (np.sum(dz, axis=0, keepdims=True)).T      # (l1 x 1)
        return [dz, dW, db]


    def predict(self, X):
        no_of_layers = len(self.layers) - 1
        a = []
        a.append(X)

        for i in range(no_of_layers):
            a.append(self.forward(a[i], self.dict['W' + str(i+1)], self.dict['b' + str(i+1)], activation=self.activations[i]))
        return a[-1]


    def normalize(self, x):
        mean = np.mean(x, axis=0)
        deviation = np.amax(x, axis=0) - np.amin(x, axis=0)
        return (x - mean)/deviation


    def confusion_matrix(self, y, y_pred):
        cm = np.zeros(shape=(2,2))
        for a, p in zip(y, y_pred):
            cm[int(a),int(p)] += 1
        return cm.ravel()


    def scores(self, y, y_pred):
        score = {}
        y_pred[y_pred == np.amax(y_pred, axis=1).reshape(-1,1)] = 1
        y_pred[y_pred < np.amax(y_pred, axis=1).reshape(-1,1)] = 0

        tn = 0
        fp = 0
        fn = 0
        tp = 0
        for i in range(y.shape[1]):
            tn_class, fp_class, fn_class, tp_class = self.confusion_matrix(y[:,i], y_pred[:,i])
            tp += tp_class
            tn += tn_class
            fp += fp_class
            fn += fn_class

        # Sensitivity, hit rate, recall, or true positive rate
        score['sensitivity'] = tp/(tp+fn)
        # Specificity or true negative rate
        score['specificity'] = tn/(tn+fp)
        # Precision or positive predictive value
        score['precision'] = tp/(tp+fp)
        # Negative predictive value
        score['npv'] = tn/(tn+fn)
        # Fall out or false positive rate
        score['fpr'] = fp/(fp+tn)
        # False negative rate
        score['fnr'] = fn/(tp+fn)
        # False discovery rate
        score['fdr'] = fp/(tp+fp)
        # Overall accuracy
        score['accuracy'] = (tp+tn)/(tp+fp+fn+tn)

        return score
