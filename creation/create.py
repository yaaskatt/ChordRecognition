from processing import datawork
from creation import pickle_data, train
from processing.paths import Path

#prep.create_references()
#prep.prepare_audio()

#pickle_data.create_beat_training_data()
#pickle_data.create_chords_training_data()

#chromas, refs, chords = datawork.get(Path.Pickle.chords_data)
beat_chromas, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beats_data)
out_exp_categ = datawork.get_categorical(beat_chords)
train.train_beat_classifier_model(beat_chromas, out_exp_categ)
"""""
y_grouper = datawork.get(Path.Pickle.beats_data)
x1_grouper = chromas[0:len(chromas) - 1]
x2_grouper = chromas[1:len(chromas)]

train.train_grouper_model(x1_grouper, x2_grouper, y_grouper)
train.train_classifier_model(chromas, out_exp_categ)
"""""


