from processing import datawork
from creation import pickle_data, train
from processing.paths import Path
import numpy as np
from sklearn.metrics import confusion_matrix
from usage import models
from creation import prep

note_name = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])



#prep.create_references()
#prep.prepare_audio()

#pickle_data.create_frame_training_data()
#pickle_data.create_beat_training_data()
#pickle_data.create_chords_training_data()

#chromas, refs, chords = datawork.get(Path.Pickle.chord_data)

beat_chromas, beat_chords, beats, chord_changes = datawork.get(Path.Pickle.beat_data)

#beat_chords_cat = datawork.get_categorical(beat_chords)
#train.train_beat_classifier_model(beat_chromas, beat_chords_cat)
#y_pred = models.classify(beat_chromas, Path.beatClassifier)

#pickle_data.create_sequencer_training_data()
seq_chords, seq_changes = datawork.get(Path.Pickle.sequencer_data)
train.train_forward_sequencer_model(datawork.get_int_array(seq_chords), datawork.get_int(beat_chords), seq_changes, 25)



#chord_changes = models.group(beat_chromas)

#train.train_sequence_model(y_pred, np.insert(chord_changes, 0, 1), beat_chords_cat)

#frame_chromas, frame_refs, frame_chords = datawork.get(Path.Pickle.frame_data)

#x1_grouper = beat_chromas[0:len(beat_chromas) - 1]
#x2_grouper = beat_chromas[1:len(beat_chromas)]

#train.train_grouper_model(x1_grouper, x2_grouper, chord_changes)
#train.train_frame_classifier_model(frame_chromas, datawork.get_categorical(frame_chords))
#train.train_chord_classifier_model(chromas, datawork.get_categorical(chords))





"""""
y_grouper = datawork.get(Path.Pickle.beats_data)
x1_grouper = chromas[0:len(chromas) - 1]
x2_grouper = chromas[1:len(chromas)]

train.train_grouper_model(x1_grouper, x2_grouper, y_grouper)
train.train_classifier_model(chromas, out_exp_categ)

conf_mx = confusion_matrix(beat_chords,
                           datawork.get_noncategorical(y_pred)[0],
                           labels=np.unique(beat_chords))
"""""


