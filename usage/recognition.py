from processing import datawork
from usage import models
from processing.paths import Path
import numpy as np

def getChords(filepath):
    chroma = datawork.get_chromagram(filepath)
    beats = datawork.get_beats(filepath)
    beat_chroma = datawork.reduceAll(np.split(chroma, beats, axis=1), 1)

    #frame_cat = models.classify(frame_chroma, Path.frameClassifier)
    beat_cat = models.classify(beat_chroma, Path.beatClassifier)
    chord_changes = models.group(beat_chroma)
    chord_changes[chord_changes >= 0.4] = 1
    chord_changes[chord_changes < 0.4] = 0

    change_indicators = chord_changes.astype(int) * beats
    grouped_beats = change_indicators[change_indicators != 0]
    grouped_chromas = datawork.reduceAll(np.split(chroma, grouped_beats, axis=1), 1)

    chord_cat = models.classify(grouped_chromas, Path.chordClassifier)

    chords_per_beat = [chord_cat[0]]

    chordNum = 0
    for i in range(len(change_indicators)):
        if change_indicators[i] != 0:
            chordNum += 1
        chords_per_beat.append(chord_cat[chordNum])

    #frame_chords, frame_accuracy = datawork.get_noncategorical(frame_cat)
    beat_chords, beat_accuracy = datawork.get_noncategorical(beat_cat)
    chord_chords, chord_accuracy = datawork.get_noncategorical(np.array(chords_per_beat))

    print("beat chord =", beat_chords[0], "accuracy =", beat_accuracy[0], "chord =", chord_chords[0], "accuracy =", chord_accuracy[0])
    for i in range(1, len(beat_chords)):
        print("beat chord =", beat_chords[i], "accuracy =", beat_accuracy[i], "chord =", chord_chords[i], "accuracy =",
              chord_accuracy[i], "chord changed =", chord_changes[i-1])



#def validate_chord(prev_chord, chord, changed):
