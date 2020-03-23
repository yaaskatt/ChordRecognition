from processing import datawork
from usage import models
from processing.paths import Path

def getChords(filepath):
    chroma = datawork.get_chromagram(filepath).T
    chroma = chroma.reshape(chroma.shape[0], chroma.shape[1], 1)

    chords_categorical = models.classify(chroma)
    chords, accuracy = datawork.get_noncategorical(chords_categorical)

    chord_changes = models.group(chroma[0:len(chroma) - 1], chroma[1:len(chroma)])

    print("chord =", chords[0], "accuracy=", accuracy[0])
    for i in range(1, len(chords)):
        print("chord =", chords[i], "accuracy=", accuracy[i], "chord change this=", chord_changes[i-1])

#def validate_chord(prev_chord, chord, changed):
