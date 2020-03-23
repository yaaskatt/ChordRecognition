from processing import datawork
from creation import pickle_data
import librosa
import numpy as np
from processing.paths import Path, Dir
import pandas as pd

#prep.create_references()
#songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
#pickle_data.create_chords_training_data(songSet)


songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
songChroma, chordChroma, frameChroma = [], [], []

for i in range(1):
    chordSet, audio, audioPath = pickle_data.read_audio_data(songSet, i)

    for j in range(chordSet.shape[0]):
        start, end, root, type = chordSet.iloc[j]

        if j == chordSet.shape[0] - 1:
            end = audio.duration_seconds
        chord = datawork.get_chromagram_from_audio(audio, start, end)
        chordChroma.extend(np.split(chord, chord.shape[1], axis=1))

    songChroma = datawork.get_chromagram(audioPath)
    chordChroma_np = np.array(chordChroma)
datawork.print_chromagram(songChroma)
datawork.print_chromagram(chordChroma_np.reshape(chordChroma_np.shape[0], chordChroma_np.shape[1]).T)


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

