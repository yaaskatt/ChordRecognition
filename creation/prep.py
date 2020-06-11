import os
import numpy as np
from pydub import AudioSegment
from processing import datawork
from processing.paths import Dir, Path

# Преобразование p3 в wav
def prepare_audio():
    for filename in os.listdir(Dir.audioSet):
        if filename[len(filename) - 4:len(filename)] == ".mp3":
            name = filename[0:len(filename) - 4]
            mp3 = AudioSegment.from_mp3(Dir.audioSet + filename)
            if mp3.channels > 1:
                mp3 = mp3.set_channels(1)
            mp3.export(Dir.audioSet + name + ".wav", format="wav")
            os.remove(Dir.audioSet + filename)

# Создание словарей для конвертации
def create_references(note_name, chord_type):
    dict_intToChord = {}
    dict_chordToInt = {}
    n = 0
    for i in range(len(chord_type)):
        for j in range(len(note_name)):
            dict_intToChord[n] = note_name[j] + chord_type[i]
            dict_chordToInt[note_name[j] + chord_type[i]] = n
            n += 1
    # Добавление тишины
    dict_intToChord[n] = "N"
    dict_chordToInt["N"] = n

    datawork.save(dict_intToChord, Path.Pickle.intToChord_dict)
    datawork.save(dict_chordToInt, Path.Pickle.chordToInt_dict)
    dict_noteMap = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
    datawork.save(dict_noteMap, Path.Pickle.noteMap_dict)
