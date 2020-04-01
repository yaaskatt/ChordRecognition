class Dir:
    initialReferences = "../../chromagrams_for_training/note_c_initial_references/"
    references = "../../chromagrams_for_training/chroma_refs/"
    chordSets = "../../songs_for_training/chords/"
    audioSet = "../../songs_for_training/wav/"

class Path:
    class Pickle:
        chord_data = "../pickled_data/chord_data.pickle"
        beat_data = "../pickled_data/beat_data.pickle"
        frame_data = "../pickled_data/frame_data.pickle"
        sequencer_data = "..pickled_data/sequencer_data.pickle"
        intToChord_dict = "../dict/chord_from_int.pickle"
        chordToInt_dict = "../dict/int_from_chord.pickle"
        noteMap_dict = "../dict/note_map.pickle"

    grouper = "../models/grouper.h5"
    denoiser = "../models/denoiser.h5"
    frameClassifier = "../models/frameClassifier.h5"
    beatClassifier = "../models/beatClassifier.h5"
    chordClassifier = "../models/chordClassifier.h5"
    reducer = "../models/reducer.pickle"
    song_set = "../../songs_for_training/song_set.csv"

