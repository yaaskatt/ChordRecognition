from processing import datawork
from usage import models
import numpy as np
import math

def getChords(filepath):
    # Получение вывода классификатора
    chroma = datawork.get_chromagram(filepath)

    beats = datawork.get_beats(filepath)
    beat_chroma = datawork.reduceAll(np.split(chroma, beats, axis=1), 1)

    chord_changes = models.group(beat_chroma)
    chords_cat = models.classify(beat_chroma)
    result = np.zeros(chords_cat.shape[0])

    beat_chords, beat_accuracy = datawork.get_noncategorical(chords_cat)

    # Преобразование классифицированных аккордов при помощи секвенсоров
    if len(beat_chords) >= 40:
        for i in range(5):

            for i in range(19, len(chords_cat) - 20):
                seq1 = np.append(chords_cat[i-19:i], chord_changes[i-18:i+1].reshape(19, 1), axis=1)
                seq2 = np.append(chords_cat[i+19:i:-1], chord_changes[i+19:i:-1].reshape(19,1), axis=1)
                result[i] = get_sequence_pred(seq1, seq2, chords_cat[i])
            for i in range(18, -1, -1):
                seq2 = np.append(chords_cat[i+19:i:-1], chord_changes[i+19:i:-1].reshape(19, 1), axis=1)
                result[i] = get_backward_pred(seq2, chords_cat[i])
            for i in range(len(chords_cat) - 20, len(chords_cat)):
                seq1 = np.append(chords_cat[i - 19:i], chord_changes[i - 18:i+1].reshape(19, 1), axis=1)
                result[i] = get_forward_pred(seq1, chords_cat[i])

            chords_cat = datawork.get_categorical_from_int(result)

        result_chords = datawork.get_chordNames(result)
    else:
        result_chords = datawork.get_noncategorical(chords_cat)
    for i in range(len(result_chords)):
        print("classified:", beat_chords[i], "result:", result_chords[i])

    grouped_beat_chords = []
    prevChord = -1
    k = -1

    # Объединение битов для формирования временных интервалов звучания
    for i in range(len(result_chords)):
        if prevChord == result_chords[i]:
            grouped_beat_chords[k][1] = i
        else:
            k += 1
            grouped_beat_chords.append([i, i, result_chords[i]])
            prevChord = result_chords[i]
    grouped_beat_chords = np.array(grouped_beat_chords)
    grouped_time_chords = ([datawork.get_time(np.insert(beats, 0, 0), grouped_beat_chords[:,0]),
                            datawork.get_time(np.append(beats, chroma.shape[1]), grouped_beat_chords[:,1]),
                            grouped_beat_chords[:,2]])
    grouped_time_chords = np.array(grouped_time_chords).T
    return grouped_time_chords

# Получение вывода с использованием обоих секвенсоров
def get_sequence_pred(seq1, seq2, classified):
    chord_prob1 = models.predict(seq1, 'f')
    chord_prob2 = models.predict(seq2, 'b')

    chord_prob1 = (-chord_prob1).argsort()[:8]
    chord_prob2 = (-chord_prob2).argsort()[:8]
    classified = (-classified).argsort()[:2]

    i, j = 0, 0
    # Исключение маловероятных аккордов
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

    # Расчёт вывода на основе выхода классификатора и секвенсоров
    change1, change2 = seq1[18][25], seq2[18][25]

    if classified[0] in chord_prob1:
        return classified[0]
    if classified[1] in chord_prob1:
        return classified[1]
    if change2 < change1:
        return probable[chord_prob2[1]]
    if change1 < change2:
        return probable[chord_prob1[0]]
    return probable[0]

# Получение вывода с использованием обратного секвенсора
def get_backward_pred(seq2, classified):
    chord_prob2 = models.predict(seq2, 'b')
    chord_prob2 = (-chord_prob2).argsort()[:8]
    classified = (-classified).argsort()[:2]

    if classified[0] in chord_prob2:
        return classified[0]
    if classified[1] in chord_prob2:
        return classified[1]
    return chord_prob2[0]

# Получение вывода с использованием прямого секвенсора
def get_forward_pred(seq1, classified):
    chord_prob1 = models.predict(seq1, 'f')
    chord_prob1 = (-chord_prob1).argsort()[:8]
    classified = (-classified).argsort()[:2]

    if classified[0] in chord_prob1:
        return classified[0]
    if classified[1] in chord_prob1:
        return classified[1]
    return chord_prob1[0]

