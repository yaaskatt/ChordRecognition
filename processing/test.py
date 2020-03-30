from processing import datawork
from creation import pickle_data
import librosa
import numpy as np
from processing.paths import Path, Dir
from usage import models
import pandas as pd


#x1_grouper, x2_grouper, y_grouper = datawork.get(Path.Pickle.beats_data)

#chromas, refs, chords = datawork.get(Path.Pickle.chords_data)[3:6]

#out_exp_categ = datawork.get_categorical(chords)
#train.train_classifier_model(chromas, out_exp_categ)

beat_chroma, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beat_data)

pred = models.group(beat_chroma)

for i in range(len(chord_changes)):
    print("actual:", chord_changes[i], "pred:", pred[i])

"""""
out_categ = models.classify(chromas[0:10000])
out_noncateg, confidence = datawork.get_noncategorical(out_categ)

for i in range(len(out_noncateg)):
    print("Ожидаемый аккорд: ", chords[i],
          "    Полученный аккорд: ", out_noncateg[i],
          "    Уверенность: ", confidence[i], sep="")
          
"""""

