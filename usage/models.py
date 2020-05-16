from keras.models import load_model
from processing.paths import Path
import numpy as np
from processing import datawork


def denoise(x):
    model = load_model(Path.denoiser)
    y = model.predict(x.reshape(x.shape[0], x.shape[1], x.shape[2], 1))
    return y.reshape(y.shape[0], y.shape[1], y.shape[2])

def classify(x, classifierPath):
    model = load_model(classifierPath)
    num, rows, cols = x.shape[0], x.shape[1], x.shape[2]
    y = model.predict(x.reshape(num, rows, cols, 1))
    return y

def group(x):
    x1 = x[0:len(x) - 1]
    x2 = x[1:len(x)]
    model = load_model(Path.grouper, compile=False)
    y = model.predict([x1, x2])
    return np.insert(np.append(y.reshape(y.shape[0] * y.shape[1]), 0), 0, 0)

def predict(x, model):
    x = x.reshape(1, x.shape[0], x.shape[1])
    y = model.predict(x)
    return y.reshape(y.shape[1])

def compute_accuracy(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

# TEST
'''
x1, x2, y_true = datawork.get(Path.Pickle.beats_data)
y_pred = group(x1, x2)
acc = compute_accuracy(y_true, y_pred)
print(acc)
for i in range(len(x1)):
    print ("same =", y_true[i], "result =", y_pred[i])
'''