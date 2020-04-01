from processing import datawork
from creation import pickle_data
import librosa
import numpy as np
from processing.paths import Path, Dir
from usage import models
import pandas as pd
from keras.models import load_model

model = load_model(Path.beatClassifier)
model.summary()

