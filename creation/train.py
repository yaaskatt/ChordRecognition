from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Conv1D, Conv2D, Flatten, Dropout, Lambda, Reshape, MaxPooling1D, LSTM, TimeDistributed
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
from keras.optimizers import RMSprop, Adam
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from processing.paths import Path
from sklearn.metrics import confusion_matrix
import pandas as pd
from processing import datawork

def train_beat_classifier_model(x, y):
    resample = SMOTE()
    x = x.reshape(x.shape[0], x.shape[1])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])

    x_train, y_train = resample.fit_resample(x_train, y_train)
    x_train = np.array(np.split(x_train.T, len(x_train), axis=1))
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)

    input_shape = (x_train.shape[1:])

    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, activation='relu', padding='same', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Conv2D(96, kernel_size=3, activation='relu', padding='same'))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, kernel_size=1, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2054, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(y.shape[1], activation='softmax'))

    adam = Adam()
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(x_train, y_train,
              epochs=30,
              batch_size=1000,
              validation_data=(x_test, y_test))
    model.summary()
    y_actu = pd.Series(datawork.get_noncategorical(y_test)[0], name='Actual')
    y_pred = pd.Series(datawork.get_noncategorical(model.predict(x_test))[0], name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    print(df_confusion)
    model.save(Path.beatClassifier)

# Базовая нейронная сеть для сиамской сети
def create_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
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
    resample = SMOTETomek()
    x_resampled, y_resampled = resample.fit_resample(x, y)
    x1_resampled, x2_resampled = np.split(x_resampled, 2, axis=1)
    x1_resampled = x1_resampled.reshape(x1_resampled.shape[0], x1.shape[1], x1.shape[2])
    x2_resampled = x2_resampled.reshape(x2_resampled.shape[0], x2.shape[1], x2.shape[2])

    x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1_resampled, x2_resampled,
                                                                             y_resampled, test_size=0.2)

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
              epochs=400,
              validation_data=([x1_test, x2_test], y_test))
    model.summary()
    model.save(Path.grouper)

def train_forward_sequencer_model(chords_true, changes, classes_num):
    x, y = [], []
    for m in range(20, len(chords_true)):
        x.append(np.append(chords_true[m - 20:m - 1], (changes[m-20:m-1]).reshape(19, 1), axis=1))
        y.append(chords_true[m - 1])
    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    model = Sequential()
    model.add(LSTM(256, input_shape=x_train.shape[1:], return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(classes_num, activation='softmax'))
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=10,
              verbose=2,
              validation_data=(x_test, y_test),
              )
    model.summary()
    model.save(Path.sequencer_fw)


def train_backward_sequencer_model(chords_true, changes, classes_num):
    x, y = [], []
    for m in range(0, len(chords_true) - 20):
        x.append(np.append(chords_true[m + 19:m:-1], (changes[m:m+19][::-1]).reshape(19, 1), axis=1))
        y.append(chords_true[m])
    x = np.array(x)
    y = np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    model = Sequential()
    model.add(LSTM(256, input_shape=x_train.shape[1:], return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(classes_num, activation='softmax'))
    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(x_train, y_train,
              epochs=10,
              batch_size=10,
              verbose=2,
              validation_data=(x_test, y_test),
              )

    model.summary()
    model.save(Path.sequencer_bw)