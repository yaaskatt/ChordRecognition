from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Conv2D, Flatten, Dropout, Lambda, Reshape
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from keras.optimizers import RMSprop
from imblearn.over_sampling import SMOTE
from processing.paths import Path
from processing import datawork


def autoenc(input_shape):
    # Вход
    x = Input(name='inputs', shape=input_shape, dtype='float32')
    flat_x = Flatten()(x)
    # Кодировщик
    enc = Dense(input_shape[0], activation='relu', name='encoder')(flat_x)
    # Декодер
    dec = Dense(input_shape[0] * input_shape[1], activation='sigmoid', name='decoder')(enc)
    dec = Reshape((input_shape[0], input_shape[1], input_shape[2]))(dec)
    Model(inputs=x, outputs=dec).summary()
    return Model(inputs=x, outputs=dec)

def train_denoiser_model(x, y):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
    input_shape = (x.shape[1:])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = autoenc(input_shape)
    model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=2056,
              shuffle=True, validation_data=(x_test, y_test))
    model.save(Path.denoiser)


def train_classifier_model(x, y):
    x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
    input_shape = (x.shape[1:])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = Sequential()
    model.add(Conv2D(48, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(y.shape[1], activation='softmax'))

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=130,
              batch_size=1000,
              shuffle=True,
              validation_data=(x_test, y_test))
    model.save(Path.classifier)

def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation="sigmoid")(x)
    return Model(input, x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def train_grouper_model(x1, x2, y):
    input_shape = (x1.shape[1:])

    x = np.concatenate((x1.reshape(x1.shape[0], x1.shape[1] * x1.shape[2]),
                              x2.reshape(x2.shape[0], x2.shape[1] * x2.shape[2])), axis=1)
    oversample = SMOTE()
    x_oversampled, y_oversampled = oversample.fit_resample(x, y)
    x1_oversampled, x2_oversampled = np.split(x_oversampled, 2, axis=1)
    x1_oversampled = x1_oversampled.reshape(x1_oversampled.shape[0], x1.shape[1], x1.shape[2])
    x2_oversampled = x2_oversampled.reshape(x2_oversampled.shape[0], x2.shape[1], x2.shape[2])

    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_oversampled, x2_oversampled,
                                                                             y_oversampled, test_size=0.3)

    base_network = create_base_network(input_shape)
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    processed1 = base_network(input1)
    processed2 = base_network(input2)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed1, processed2])

    model = Model([input1, input2], distance)

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
    model.fit([x1_train, x2_train], y_train,
              batch_size=2056,
              epochs=70,
              validation_data=([x1_test, x2_test], y_test))
    model.save(Path.grouper)


#x1_grouper, x2_grouper, y_grouper = datawork.get(Path.Pickle.beats_data)
#train_grouper_model(x1_grouper, x2_grouper, y_grouper)













































