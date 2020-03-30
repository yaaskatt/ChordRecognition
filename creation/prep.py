import os
import numpy as np
from pydub import AudioSegment
from processing import datawork
from processing.paths import Dir, Path


# Генерирование эталонных хромаграмм, присвоение аккордам цифр, установление соответствий между аккордами с # и b
def create_references():
    note_name = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
    n = 0
    dict_intToChord = {}
    dict_chordToInt = {}
    for filename in os.listdir(Dir.initialReferences):
        if filename[len(filename)-3:len(filename)] != "txt":
            continue
        root = filename[0]
        type = ""
        # Если в аккорде больше одного символа
        if len(filename) > 5:
            if filename[1] == 'b':
                root += 'b'
                # Если в аккорде больше двух символов
                if len(filename) > 6:
                    type = filename[2:len(filename)-4]
            else:
                type = filename[1:len(filename)-4]

        chroma = datawork.read_matrix(Dir.initialReferences + "/" + filename)
        datawork.save(chroma, Dir.references + "/" + root + "/" + filename[0:len(filename) - 4] + ".pickle")
        dict_intToChord[n] = root + type
        dict_chordToInt[root + type] = n
        n += 1  # для словаря

        for i in range(1, len(note_name)):
            # Добавление пар (аккорд - цифра) в словари
            dict_intToChord[n] = note_name[i] + type
            dict_chordToInt[note_name[i] + type] = n

            # циклическая перестановка для создания хромаграммы каждого аккорда
            chroma = np.roll(chroma, 1, axis=0)
            datawork.save(chroma, Dir.references + "/" + note_name[i] + "/" + note_name[i] + type + ".pickle")
            n += 1

    # Тишина
    dict_intToChord[n] = "N"
    dict_chordToInt["N"] = n
    datawork.save(np.zeros(12), Dir.references + "/N/N.pickle")

    datawork.save(dict_intToChord, Path.Pickle.intToChord_dict)
    datawork.save(dict_chordToInt, Path.Pickle.chordToInt_dict)
    dict_noteMap = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
    datawork.save(dict_noteMap, Path.Pickle.noteMap_dict)


# Преобразование аудио в нужный формат
def prepare_audio():
    for filename in os.listdir(Dir.audioSet):
        if filename[len(filename) - 4:len(filename)] == ".mp3":
            name = filename[0:len(filename) - 4]
            mp3 = AudioSegment.from_mp3(Dir.audioSet + filename)
            if mp3.channels > 1:
                mp3.set_channels(1)
            mp3.export(Dir.audioSet + name + ".wav", format="wav")
            os.remove(Dir.audioSet + filename)
