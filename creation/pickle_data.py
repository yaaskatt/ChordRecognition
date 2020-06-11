import pandas as pd
import numpy as np
from processing import datawork
from processing.paths import Dir
from processing.paths import Path

# Чтение строки из файла с аккордами
def read_audio_data(songSet, index):
    audioFile, chordSet_file = songSet.iloc[index]
    print(audioFile)
    chordSet = pd.read_csv(Dir.chordSets + "/" + chordSet_file, sep=";", encoding="UTF-8", keep_default_na=False)
    audioPath = Dir.audioSet + "/" + audioFile
    audio = AudioSegment.from_wav(audioPath)
    return chordSet, audio, audioPath

# Создание обучающих данных
def create_beat_training_data(note_name, chord_type):
    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    noteMap_dict = datawork.get(Path.Pickle.noteMap_dict)
    chord_changes = []
    beat_chroma, beat_chords = [], []
    for k in range(songSet.shape[0]):
        print("song №", k+1, sep="")
        chordSet, audio, audioPath = read_audio_data(songSet, k)
        # Создание хромаграммы песни
        songChroma = datawork.get_chromagram(audioPath)
        datawork.print_chromagram(songChroma)
        beats = datawork.get_beats(audioPath)
        start = 0
        j, m = 0, 0
        # Обработка битов
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

            chord_start, chord_end, root, type = chordSet.iloc[j]

            # Определение битов, на которых сменяются аккорды
            if k == 0 and i == 0:
                print()
            elif i == 0:
                chord_changes.append(0)
                print("same=", chord_changes[len(chord_changes) - 1])
            else:
                if start_time > chord_start and end_time < chord_end:
                    chord_changes.append(1)
                elif end_time > chord_end:
                    if chord_end - start_time < end_time - chord_end or \
                            (end_time - chord_end > 2 and (chord_end - start_time) - (end_time - chord_end) < 1):
                        chord_changes.append(0)
                        j += 1
                        root, type = chordSet.iloc[j][2:4]
                    else:
                        chord_changes.append(1)
                print("same=", chord_changes[len(chord_changes) - 1])
            if root not in note_name or type not in chord_type:
                return
            if root in noteMap_dict:
                root = noteMap_dict[root]
            beat_chords.append(root + type)
            beat_chroma.append(datawork.reduce(songChroma[:, start:end], 1))
            start = end

    #Сохранение обучающих данных
    datawork.save((np.array(beat_chroma), np.array(beat_chords), np.array(beats), np.array(chord_changes)),
                  Path.Pickle.beat_data)








