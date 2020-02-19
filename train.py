from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential


def autoenc(input_shape):
    # Вход
    x = Input(name='inputs', shape=input_shape, dtype='float32')
    o = x
    # Кодировщик
    enc = Dense(12, activation='relu', name='encoder')(o)
    # Декодер
    dec = Dense(input_shape[0], activation='sigmoid', name='decoder')(enc)
    Model(inputs=x, outputs=dec).summary()
    return Model(inputs=x, outputs=dec)

def train_denoiser_model(x, y, model_path):
    rows, cols = x.shape[1], x.shape[2]
    input_shape = (rows * cols,)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = x_train.reshape(x_train.shape[0], rows * cols)
    y_train = y_train.reshape(y_train.shape[0], rows * cols)
    x_test = x_test.reshape(x_test.shape[0], rows * cols)
    y_test = y_test.reshape(y_test.shape[0], rows * cols)

    batch_size = x_train.shape[0]
    epochs = 1000
    model = autoenc(input_shape)
    model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                shuffle=True, validation_data=(x_test, y_test))
    model.save(model_path)


def train_classifier_model(x, y, model_path):
    rows, cols = x.shape[1], x.shape[2]
    input_shape = (rows, cols, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = x_train.reshape(x_train.shape[0], rows, cols, 1)
    x_test = x_test.reshape(x_test.shape[0], rows, cols, 1)

    batch_size = x_train.shape[0]
    epochs = 500
    model = Sequential()
    model.add(Conv2D(48, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              shuffle=True, validation_data=(x_test, y_test))
    model.save(model_path)









































