from pydub import AudioSegment
import librosa
import datawork
import pandas as pd
import numpy as np
import pickle_data
from paths import Path
import models

"""""
songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
chroma = []
for k in range(1):
    print("song №", k+1, sep="")
    chordSet, audio, audioPath = pickle_data.read_audio_data(songSet, k)
    beats = datawork.get_beats(audioPath)

    start = 0
    j = 0
    for i in range(len(beats)):
        if i != 0:
            start = beats[i - 1]
        if i != len(beats) - 1:
            end = beats[i]
        else:
            end = audio.duration_seconds
        print("i =", i, "start =", start, "end =", end)
        chroma.append(datawork.get_chromagram_from_audio(audio, start, end))
        start = end

datawork.save(chroma, Path.Pickle.beats_data)
"""""
chroma = datawork.get(Path.Pickle.beats_data)
out_categ = models.classify(datawork.reduceAll(chroma, 5))

# Выход классификатора в виде НАЗВАНИЯ АККОРДА
out_noncateg, confidence = datawork.get_noncategorical(out_categ)
for i in range(len(out_noncateg)):
    print("    Аккорд: ", out_noncateg[i],
          "    Уверенность: ", confidence[i], sep="")




