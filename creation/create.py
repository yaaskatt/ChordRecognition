from processing import datawork
from creation import pickle_data, train
from processing.paths import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from usage import models
from creation import prep

note_name = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])

prep.create_references()
#prep.prepare_audio()
pickle_data.create_beat_training_data()
chromas, refs, chords = datawork.get(Path.Pickle.chord_data)
beat_chromas, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beat_data)
beat_chords_cat = datawork.get_categorical(beat_chords)
train.train_beat_classifier_model(beat_chromas, beat_chords_cat)

x1_grouper = beat_chromas[0:len(beat_chromas) - 1]
x2_grouper = beat_chromas[1:len(beat_chromas)]
train.train_grouper_model(x1_grouper, x2_grouper, chord_changes)

pickle_data.create_sequencer_training_data()
seq_chords, seq_changes = datawork.get(Path.Pickle.sequencer_data)
train.train_forward_sequencer_model(seq_chords, datawork.get_categorical(beat_chords), seq_changes, 25)
train.train_backward_sequencer_model(seq_chords, datawork.get_categorical(beat_chords), seq_changes, 25)

y_pred = models.classify(beat_chords)
conf_mx = confusion_matrix(beat_chords,
                           datawork.get_noncategorical(y_pred)[0],
                           labels=np.unique(beat_chords))


