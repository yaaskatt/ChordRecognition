from processing import datawork
from creation import pickle_data, train
from processing.paths import Path
import numpy as np
from creation import prep

note_name = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])



#prep.create_references()
#prep.prepare_audio()

#pickle_data.create_frame_training_data()
#pickle_data.create_beat_training_data()
#pickle_data.create_chords_training_data()

#chromas, refs, chords = datawork.get(Path.Pickle.chord_data)



beat_chromas, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beat_data)
beat_chords_cat = datawork.get_categorical(beat_chords)

additional_beat_chromas = []
additional_beat_chords_cat = []

for i in range(1, 12):
    additional_beat_chromas.extend(np.roll(beat_chromas.reshape(beat_chromas.shape[0],
                                                                beat_chromas.shape[1]), i, axis=1))
    additional_beat_chords_cat.extend(np.roll(beat_chords_cat.reshape(beat_chords_cat.shape[0],
                                                                      beat_chords_cat.shape[1]), i, axis=1))

beat_chromas = np.concatenate((beat_chromas, np.array(np.split(np.array(additional_beat_chromas).T, len(additional_beat_chromas), axis=1))), axis=0)

temp = np.array(beat_chords_cat)
beat_chords_cat = np.concatenate((beat_chords_cat, np.array(beat_chords_cat)), axis=0)


#frame_chromas, frame_refs, frame_chords = datawork.get(Path.Pickle.frame_data)

#x1_grouper = beat_chromas[0:len(beat_chromas) - 1]
#x2_grouper = beat_chromas[1:len(beat_chromas)]

#train.train_grouper_model(x1_grouper, x2_grouper, chord_changes)
#train.train_frame_classifier_model(frame_chromas, datawork.get_categorical(frame_chords))
train.train_beat_classifier_model(beat_chromas, datawork.get_categorical(beat_chords))
#train.train_chord_classifier_model(chromas, datawork.get_categorical(chords))
"""""
y_grouper = datawork.get(Path.Pickle.beats_data)
x1_grouper = chromas[0:len(chromas) - 1]
x2_grouper = chromas[1:len(chromas)]

train.train_grouper_model(x1_grouper, x2_grouper, y_grouper)
train.train_classifier_model(chromas, out_exp_categ)
"""""


