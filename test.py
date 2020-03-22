from pydub import AudioSegment
import librosa
import datawork
import pandas as pd
import numpy as np
import pickle_data
from paths import Dir, Path
import prep
import train
import models

#prep.create_references()
#songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
#pickle_data.create_chords_training_data(songSet)

chromas, refs = datawork.get(Path.Pickle.chords_data)[3:5]
train.train_denoiser_model(chromas, refs)

#test_chromas = chromas[0:15]
#test_refs = refs[0:15]
#denoised = models.denoise(test_chromas)

#x1_grouper, x2_grouper, y_grouper = datawork.get(Path.Pickle.beats_data)
#train.train_grouper_model(x1_grouper, x2_grouper, y_grouper)

#chromas, refs, chords = datawork.get(Path.Pickle.chords_data)[3:6]

#out_exp_categ = datawork.get_categorical(chords)
#train.train_classifier_model(chromas, out_exp_categ)

"""""
out_categ = models.classify(chromas[0:10000])
out_noncateg, confidence = datawork.get_noncategorical(out_categ)

for i in range(len(out_noncateg)):
    print("Ожидаемый аккорд: ", chords[i],
          "    Полученный аккорд: ", out_noncateg[i],
          "    Уверенность: ", confidence[i], sep="")
          
"""""

