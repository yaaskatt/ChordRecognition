from processing import datawork
from creation import pickle_data
import librosa
import numpy as np
from processing.paths import Path, Dir
from usage import models
import pandas as pd
from keras.models import load_model

model = load_model(Path.beatClassifier)
beat_chromas, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beat_data)

beat_chords_cat = datawork.get_categorical(beat_chords)
model.summary()
acc = model.evaluate(beat_chromas.reshape(beat_chromas.shape[0], beat_chromas.shape[1], 1, 1), beat_chords_cat)

