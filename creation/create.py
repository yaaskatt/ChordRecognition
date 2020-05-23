from processing import datawork
from creation import pickle_data, train
from processing.paths import Path
import numpy as np
from creation import prep

note_name = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
chord_type = np.array(['', 'm'])
prep.create_references(note_name, chord_type)
prep.prepare_audio()
pickle_data.create_beat_training_data(note_name, chord_type)

beat_chromas, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beat_data)
beat_chords_cat = datawork.get_categorical(beat_chords)
pickle_data.create_sequencer_training_data()
seq_chords, seq_changes = datawork.get(Path.Pickle.sequencer_data)
train.train_forward_sequencer_model(datawork.get_categorical(beat_chords), seq_changes, 25)
"""""
train.train_beat_classifier_model(beat_chromas, beat_chords_cat)

x1_grouper = beat_chromas[0:len(beat_chromas) - 1]
x2_grouper = beat_chromas[1:len(beat_chromas)]
train.train_grouper_model(x1_grouper, x2_grouper, chord_changes)

pickle_data.create_sequencer_training_data()
seq_chords, seq_changes = datawork.get(Path.Pickle.sequencer_data)
train.train_forward_sequencer_model(seq_chords, datawork.get_categorical(beat_chords), seq_changes, 25)
train.train_backward_sequencer_model(seq_chords, datawork.get_categorical(beat_chords), seq_changes, 25)
"""""




