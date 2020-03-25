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
    noteMap_dict = datawork.get(Path.Pickle.noteMap_dict)
    chord_changes = []
    beat_chroma, beat_chords = [], []
    for k in range(songSet.shape[0]):
        print("song №", k+1, sep="")
        chordSet, audio, audioPath = read_audio_data(songSet, k)
        songChroma = datawork.get_chromagram(audioPath)
        beats = datawork.get_beats(audioPath)
        start = 0
        j, m = 0, 0
        for i in range(len(beats)):
            if i != 0:
                start = beats[i - 1]
            if i != len(beats) - 1:
                end = beats[i]
            else:
                end = songChroma.shape[1]
            print("i =", i, "start =", start, "end =", end, end=" ")

            start_time = librosa.frames_to_time(start)
            end_time = librosa.frames_to_time(end)
            if i == 0:
                print()
            chord_start, chord_end, root, type = chordSet.iloc[j]
            if root in noteMap_dict:
                root = noteMap_dict[root]
            if i != 0:

                if start_time > chord_start and end_time < chord_end:
                    chord_changes.append(1)
                elif end_time > chord_end:
                    if chord_end - start_time < end_time - chord_end or \
                            (end_time - chord_end > 2 and (chord_end - start_time) - (end_time - chord_end) < 1):
                        chord_changes.append(0)
                        j += 1
                    else:
                        chord_changes.append(1)
                print("same=", chord_changes[len(chord_changes) - 1])

            beat_chords.append(root + type)
            beat_chroma.append(datawork.reduce(songChroma[:, start:end], 1))
            start = end


    datawork.save((np.array(beat_chroma), np.array(beat_chords), np.array(beats), np.array(chord_changes)),
                  Path.Pickle.beats_data)


# Чтение данных из датасетов
def create_chords_training_data():

    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    noteMap_dict = datawork.get(Path.Pickle.noteMap_dict)
    chroma, chromaRefs, chords = [], [], []

    for i in range(songSet.shape[0]):
        chordSet, audio, audioPath = read_audio_data(songSet, i)
        songChroma = datawork.get_chromagram(audioPath)
        print("CHROMA CONTAINS", songChroma.shape[1], "FRAMES")

        chordNum, lastFrameAdded = 0, 0
        while chordNum < chordSet.shape[0]:
            print("chord changed")

            end, root, type = chordSet.iloc[chordNum][1:4]
            endFrame = librosa.time_to_frames(end)
            if chordNum == chordSet.shape[0] - 1:
                endFrame = songChroma.shape[1]
            if root in noteMap_dict:
                root = noteMap_dict[root]

            reference = np.array(datawork.get(Dir.references + "/" + root + "/" + root + type + ".pickle")).T
            chord = root + type

            while lastFrameAdded < songChroma.shape[1] and endFrame > lastFrameAdded:
                print("total:", songChroma.shape[1], "now:", lastFrameAdded+1, "this chord should be till", endFrame)
                chromaRefs.append(reference)
                chords.append(chord)
                lastFrameAdded += 1

            chordNum += 1

        chroma.extend(songChroma.T.reshape(songChroma.shape[1], songChroma.shape[0], 1))

        if len(chords) != len(chroma):
            print ("НЕ РАВНО")
            return
    chromaRefs_np = np.array(chromaRefs)
    chromaRefs_np = chromaRefs_np.reshape(chromaRefs_np.shape[0], chromaRefs_np.shape[1], 1)
    datawork.save((np.array(chroma), chromaRefs_np, np.array(chords)), Path.Pickle.chords_data)



