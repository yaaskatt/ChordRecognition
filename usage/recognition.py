from processing import datawork
from usage import models
from processing.paths import Path
import numpy as np
from keras.models import load_model
import math

def getChords(filepath):
    chroma = datawork.get_chromagram(filepath)
    beats = datawork.get_beats(filepath)
    beat_chroma = datawork.reduceAll(np.split(chroma, beats, axis=1), 1)

    chord_changes = models.group(beat_chroma)
    chords_cat = models.classify(beat_chroma, Path.beatClassifier)
    result = np.zeros(chords_cat.shape[0])

    model_f = load_model(Path.sequencer_fw)
    model_b = load_model(Path.sequencer_bw)

    beat_chords, beat_accuracy = datawork.get_noncategorical(chords_cat)

    for i in range(5):

        for i in range(19, len(chords_cat) - 20):
            seq1 = np.append(chords_cat[i-19:i], chord_changes[i-18:i+1].reshape(19, 1), axis=1)
            seq2 = np.append(chords_cat[i+19:i:-1], chord_changes[i+19:i:-1].reshape(19,1), axis=1)
            result[i] = get_sequence_pred(seq1, seq2, chords_cat[i], model_f, model_b)
            #print("classified:", beat_chords[i], "result:", datawork.get_chordNames(([result[i]]))[0])
        for i in range(18, -1, -1):
            seq2 = np.append(chords_cat[i+19:i:-1], chord_changes[i+19:i:-1].reshape(19, 1), axis=1)
            result[i] = get_backward_pred(seq2, chords_cat[i], model_b)
            #print("classified:", beat_chords[i], "result:", datawork.get_chordNames(([result[i]]))[0])
        for i in range(len(chords_cat) - 20, len(chords_cat)):
            seq1 = np.append(chords_cat[i - 19:i], chord_changes[i - 18:i+1].reshape(19, 1), axis=1)
            result[i] = get_forward_pred(seq1, chords_cat[i], model_f)
            #print("classified:", beat_chords[i], "result:", datawork.get_chordNames(([result[i]]))[0])

        chords_cat = datawork.get_categorical_from_int(result)


    result_chords = datawork.get_chordNames(result)
    for i in range(len(result_chords)):
        print("classified:", beat_chords[i], "result:", result_chords[i])

    grouped_beat_chords = []
    prevChord = -1
    k = -1
    for i in range(len(result_chords)):
        if i == 39:
            print()
        if prevChord == result_chords[i]:
            grouped_beat_chords[k][1] = i
        else:
            k += 1
            grouped_beat_chords.append([i, i, result_chords[i]])
            prevChord = result_chords[i]
    grouped_beat_chords = np.array(grouped_beat_chords)
    print(grouped_beat_chords)
    grouped_time_chords = ([datawork.get_time(np.insert(beats, 0, 0), grouped_beat_chords[:,0]),
                            datawork.get_time(np.append(beats, chroma.shape[1]), grouped_beat_chords[:,1]),
                            grouped_beat_chords[:,2]])
    grouped_time_chords = np.array(grouped_time_chords).T
    print(grouped_time_chords)
    return grouped_time_chords


def get_sequence_pred(seq1, seq2, classified, model_f, model_b):
    chord_prob1 = models.predict(seq1, model_f)
    chord_prob2 = models.predict(seq2, model_b)

    chord_prob1 = (-chord_prob1).argsort()[:8]
    chord_prob2 = (-chord_prob2).argsort()[:8]
    classified = (-classified).argsort()[:2]

    i, j = 0, 0
    while True:
        if i < chord_prob1.shape[0] and chord_prob1[i] not in chord_prob2:
            chord_prob1 = np.delete(chord_prob1, i)
        else:
            i += 1
        if j < chord_prob2.shape[0] and chord_prob2[j] not in chord_prob1:
            chord_prob2 = np.delete(chord_prob2, j)
        else:
            j += 1
        if i >= chord_prob1.shape[0] and j >= chord_prob2.shape[0]:
            break

    probable = []

    for i in range(chord_prob1.shape[0]):
        index = math.floor((i + np.argwhere(chord_prob2 == chord_prob1[i])) / 2)
        if index > len(probable):
            index = len(probable)
        probable.insert(index, chord_prob1[i])

    probable = np.array(probable)
    change1, change2 = seq1[18][25], seq2[18][25]

    #print(classified, chord_prob1, chord_prob2, probable)

    if classified[0] in probable:
        return classified[0]
    if classified[1] in probable:
        return classified[1]
    place1 = np.argwhere(probable == chord_prob1[0])
    place2 = np.argwhere(probable == chord_prob2[0])
    if change2 < change1:
        return probable[place2]
    if change1 < change2:
        return probable[place1]
    return probable[0]

def get_backward_pred(seq2, classified, model_b):
    chord_prob2 = models.predict(seq2, model_b)
    chord_prob2 = (-chord_prob2).argsort()[:8]
    classified = (-classified).argsort()[:2]

    #print(classified, print(chord_prob2))

    if classified[0] in chord_prob2:
        return classified[0]
    if classified[1] in chord_prob2:
        return classified[1]
    return chord_prob2[0]

def get_forward_pred(seq1, classified, model_f):
    chord_prob1 = models.predict(seq1, model_f)
    chord_prob1 = (-chord_prob1).argsort()[:8]
    classified = (-classified).argsort()[:2]

    #print((classified, chord_prob1))

    if classified[0] in chord_prob1:
        return classified[0]
    if classified[1] in chord_prob1:
        return classified[1]
    return chord_prob1[0]

