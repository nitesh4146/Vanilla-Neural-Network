from vanilla import *
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn import datasets

# Data Prep
iris = datasets.load_iris()
X = iris.data.astype(float)
y = iris.target.reshape(-1,1)
n = X.shape[1]

# boston = datasets.load_boston()
# X = boston.data.astype(float)
# y = boston.target.reshape(-1,1)
# n = X.shape[1]

# One Hot Encoding
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray().astype(float)
out = y.shape[1]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Build Model
model = vanilla()
model.add_layer(100, input_dim=n, activation="sigmoid")
model.add_layer(60, activation='sigmoid')
model.add_layer(60, activation='sigmoid')
model.add_layer(out, activation='softmax')
model.compile(learning_rate = 0.01, loss="entropy")
loss = model.fit(X_train, y_train, nb_epoch=10, batch_size= 1000, normalize=True, regularize=False)

X_test_n = model.normalize(X_test)
print("Test Accuracy: %.3f %%\n"%(model.scores(y_test, model.predict(X_test_n))['accuracy']*100))

plt.plot(loss)
plt.show()
