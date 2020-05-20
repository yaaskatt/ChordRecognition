import os
import numpy as np
from pydub import AudioSegment
from processing import datawork
from processing.paths import Dir, Path
import re
import pandas as pd


# Генерирование эталонных хромаграмм, присвоение аккордам цифр, установление соответствий между аккордами с # и b
def create_references(note_name, chord_type):
    dict_intToChord = {}
    dict_chordToInt = {}
    n = 0
    for i in range(len(note_name)):
        for j in range(len(chord_type)):
            dict_intToChord[n] = note_name[i] + chord_type[j]
            dict_chordToInt[note_name[i] + chord_type[j]] = n
            n += 1  # для словаря

    # Тишина
    dict_intToChord[n] = "N"
    dict_chordToInt["N"] = n

    datawork.save(dict_intToChord, Path.Pickle.intToChord_dict)
    datawork.save(dict_chordToInt, Path.Pickle.chordToInt_dict)
    dict_noteMap = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
    datawork.save(dict_noteMap, Path.Pickle.noteMap_dict)

def clean_csv():
    songSet = pd.read_csv(Path.song_set, sep=";", encoding="UTF-8")
    for k in range(songSet.shape[0]):
        audioFile, chordSet_file = songSet.iloc[k]
        chordSet = pd.read_csv(Dir.chordSets + "/" + chordSet_file, sep=";", encoding="UTF-8", keep_default_na=False)
        chordSet.columns = ['start', 'end', 'root', 'type']
        #chordSet['start'] = chordSet['start'].apply(clean)
        chordSet['type'] = chordSet['type'].astype(str)
        chordSet['type'] = chordSet['type'].apply(clean_type)
        chordSet = chordSet[chordSet['start'] != '']
        chordSet.to_csv(Dir.chordSets + "/" + chordSet_file, sep=";", index=False)

def clean(x):
    try:
        x = re.sub(r'^\d+,', '', x)
        return float(x)
    except:
        return x

def clean_type(x):
    try:
        x = re.sub(r'\.0$', '', x)
        return x
    except:
        return str(x)

# Преобразование аудио в нужный формат
def prepare_audio():
    for obj in os.listdir(Dir.audioSet):
        album_folder = os.path.join(Dir.audioSet, obj)
        if os.path.isdir(album_folder):
            for filename in os.listdir(album_folder):
                if filename[len(filename) - 4:len(filename)] == ".mp3":
                    name = filename[0:len(filename) - 4]
                    name = re.sub(r'^\d+\. ', '', name)
                    mp3 = AudioSegment.from_mp3(os.path.join(album_folder, filename))
                    if mp3.channels > 1:
                        mp3 = mp3.set_channels(1)
                    mp3.export(os.path.join(album_folder, name + '.wav'), format="wav")
                    os.remove(os.path.join(album_folder, filename))
