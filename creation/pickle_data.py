import pandas as pd
from pydub import AudioSegment
import numpy as np
import librosa
from processing import datawork
from processing.paths import Dir
from processing.paths import Path


def read_audio_data(songSet, index):
    audioFile, chordSet_file = songSet.iloc[index]
    print(audioFile)
    chordSet = pd.read_csv(Dir.chordSets + "/" + chordSet_file, sep=";", encoding="UTF-8", keep_default_na=False)
    audioPath = Dir.audioSet + "/" + audioFile
    audio = AudioSegment.from_wav(audioPath)
    return chordSet, audio, audioPath


def create_beat_training_data():

    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    chord_changes = []
    for k in range(songSet.shape[0]):
        print("song №", k+1, sep="")
        chordSet, audio, audioPath = read_audio_data(songSet, k)
        beats = datawork.get_beats(audioPath)

        start = 0
        j = 0
        for i in range(len(beats)):
            if i != 0:
                start = beats[i - 1]
            if i != len(beats) - 1:
                end = beats[i]
            else:
                end = audio.duration_seconds
            print("i =", i, "start =", start, "end =", end, end=" ")
            if i == 0:
                print()
            framesNum = librosa.time_to_frames(end-start)

            if i != 0:
                chord_start, chord_end = chordSet.iloc[j][0:2]
                if start > chord_start and end < chord_end:
                    chord_changes.append(1)
                elif end > chord_end:
                    if chord_end - start < end - chord_end or \
                            (end - chord_end > 2 and (chord_end - start) - (end - chord_end) < 1):
                        chord_changes.append(0)
                        j += 1
                    else:
                        chord_changes.append(1)
                print("same=", chord_changes[len(chord_changes) - 1])
            for m in range(framesNum):
                chord_changes.append(1)

            start = end

    datawork.save(np.array(chord_changes), Path.Pickle.beats_data)


# Чтение данных из датасетов
def create_chords_training_data():

    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    noteMap_dict = datawork.get(Path.Pickle.noteMap_dict)
    chroma, chromaRefs, chords = [], [], []

    for i in range(songSet.shape[0]):
        chordSet, audio, audioPath = read_audio_data(songSet, i)
        songChroma = datawork.get_chromagram(audioPath)
        print("CHROMA CONTAINS", songChroma.shape[1], "FRAMES")
        j, k = 0, 0
        while j < chordSet.shape[0]:
            print("chord changed")
            end, root, type = chordSet.iloc[j][1:4]
            if j == chordSet.shape[0] - 1:
                end = librosa.frames_to_time(songChroma.shape[1])
            if root in noteMap_dict:
                root = noteMap_dict[root]
            reference = np.array(datawork.get(Dir.references + "/" + root + "/" + root + type + ".pickle")).T
            chord = root + type
            while k < songChroma.shape[1] and librosa.time_to_frames(end) > k:
                print("total:", songChroma.shape[1], "now:", k+1, "this chord should be till", librosa.time_to_frames(end))
                chromaRefs.append(reference)
                chords.append(chord)
                k += 1
            j += 1
        chroma.extend(songChroma.T.reshape(songChroma.shape[1], songChroma.shape[0], 1))
    chromaRefs_np = np.array(chromaRefs)
    chromaRefs_np = chromaRefs_np.reshape(chromaRefs_np.shape[0], chromaRefs_np.shape[1], 1)
    datawork.save(chroma, chromaRefs_np, np.array(chords))



