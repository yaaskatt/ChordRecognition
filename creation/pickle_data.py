import pandas as pd
from pydub import AudioSegment
import numpy as np
import datawork
from paths import Dir
from paths import Path


def read_audio_data(songSet, index):
    audioFile, chordSet_file = songSet.iloc[index]
    print(audioFile)
    chordSet = pd.read_csv(Dir.chordSets + chordSet_file, sep=";", encoding="UTF-8", keep_default_na=False)
    audioPath = Dir.audioSet + audioFile
    audio = AudioSegment.from_wav(audioPath)
    return chordSet, audio, audioPath


def create_beat_training_data(songSet):

    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    chroma1 = []
    chroma2 = []
    chord_changes = []
    for k in range(songSet.shape[0]):
        print("song №", k+1, sep="")
        songChroma = []
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
            beatChroma = datawork.get_chromagram_from_audio(audio, start, end)
            songChroma.extend(np.split(beatChroma, beatChroma.shape[1], axis=1))

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
            for m in range(beatChroma.shape[1]-1):
                chord_changes.append(1)

            start = end

        songChroma_np = np.array(songChroma)

        chroma1.extend(songChroma_np[0:len(songChroma_np) - 1])
        chroma2.extend(songChroma_np[1:len(songChroma_np)])
    datawork.save((np.array(chroma1), np.array(chroma2), np.array(chord_changes)), Path.Pickle.beats_data)


# Чтение данных из датасетов
def create_chords_training_data():

    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    noteMap_dict = datawork.get(Path.Pickle.noteMap_dict)
    chordChromas_frames, chromaRefs_frames, chords_frames = [], [], []

    for i in range(songSet.shape[0]):
        chordSet, audio = read_audio_data(songSet, i)[0:2]

        for j in range(chordSet.shape[0]):
            start, end, root, type = chordSet.iloc[j]

            if j == chordSet.shape[0] - 1:
                end = audio.duration_seconds
            if root == "N":
                continue
            # Замена нот с диезом на ноты с бемолем
            if root in noteMap_dict:
                root = noteMap_dict[root]

            chord_chroma = datawork.get_chromagram_from_audio(audio, start, end)
            reference = np.array(datawork.get(Dir.references + root + "/" + root + type + ".pickle")).T
            chord = root + type

            chordChromas_frames.extend(np.split(chord_chroma, chord_chroma.shape[1], axis=1))

            for k in range(chord_chroma.shape[1]):
                chromaRefs_frames.append(reference)
                chords_frames.append(chord)

    chromaRefs_frames_np = np.array(chromaRefs_frames)
    chromaRefs_frames_np = chromaRefs_frames_np.reshape(chromaRefs_frames_np.shape[0], chromaRefs_frames_np.shape[1], 1)

    chords_frames_np = np.array(chords_frames)

    datawork.save((np.array(chordChromas_frames), chromaRefs_frames_np, chords_frames_np), Path.Pickle.chords_data)


