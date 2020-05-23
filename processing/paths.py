class Dir:
    chordSets = "../../songs_for_training/chords_extended/"
    audioSet = "../../songs_for_training/wav/"

class Path:
    class Pickle:
        beat_data = "../pickled_data/beat_data.pickle"
        sequencer_data = "..pickled_data/sequencer_data.pickle"
        intToChord_dict = "../dict/chord_from_int.pickle"
        chordToInt_dict = "../dict/int_from_chord.pickle"
        noteMap_dict = "../dict/note_map.pickle"

    grouper = "../models/grouper.h5"
    beatClassifier = "../models/beatClassifier.h5"
    triadClassifier = "../models/triadClassifier.h5"
    sequencer_fw = "../models/sequencer_fw.h5"
    sequencer_bw = "../models/sequencer_bw.h5"
    song_set = "../../songs_for_training/song_set_extended.csv"

