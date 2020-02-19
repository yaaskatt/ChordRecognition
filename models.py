from keras.models import load_model


def denoise(modelPath, x):
    model = load_model(modelPath)
    num, rows, cols = x.shape[0], x.shape[1], x.shape[2]
    y = model.predict(x.reshape(num, rows * cols))
    return y.reshape(num, rows, cols)


def classify(modelPath, x):
    model = load_model(modelPath)
    num, rows, cols = x.shape[0], x.shape[1], x.shape[2]
    y = model.predict(x.reshape(num, rows, cols, 1))
    return y

