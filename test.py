from pydub import AudioSegment
import librosa
import datawork
import pandas as pd
import numpy as np
import pickle_data
from paths import Path
import prep
import train
import models

#prep.create_references()
#songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
#pickle_data.create_chords_training_data(songSet)

#chromas, refs = datawork.get(Path.Pickle.chords_data)[3:5]
#train.train_denoiser_model(chromas, refs)

#test_chromas = chromas[0:15]
#test_refs = refs[0:15]
#denoised = models.denoise(test_chromas)

x1_grouper, x2_grouper, y_grouper = datawork.get(Path.Pickle.beats_data)
train.train_grouper_model(models.denoise(x1_grouper), models.denoise(x2_grouper), y_grouper)



