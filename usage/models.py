from keras.models import load_model
from processing.paths import Path
import numpy as np

#Загрузка моделей
classifier = load_model(Path.beatClassifier)
grouper = load_model(Path.grouper, compile=False)
sequencer_fw = load_model(Path.sequencer_fw)
sequencer_bw = load_model(Path.sequencer_bw)

def classify(x):
    num, rows, cols = x.shape[0], x.shape[1], x.shape[2]
    y = classifier.predict(x.reshape(num, rows, cols, 1))
    return y

def group(x):
    x1 = x[0:len(x) - 1]
    x2 = x[1:len(x)]
    y = grouper.predict([x1, x2])
    return np.insert(np.append(y.reshape(y.shape[0] * y.shape[1]), 0), 0, 0)

def predict(x, direction):
    x = x.reshape(1, x.shape[0], x.shape[1])
    if direction == 'f':
        y = sequencer_fw.predict(x)
    elif direction == 'b':
        y = sequencer_bw.predict(x)
    return y.reshape(y.shape[1])