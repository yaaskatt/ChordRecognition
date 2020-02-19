import os
import numpy as np
from pydub import AudioSegment
import datawork


# Генерирование эталонных хромаграмм, присвоение аккордам цифр, установление соответствий между аккордами с # и b
def create_references(refsDir, exampleDir, dict_chordFromIntPath, dict_intFromChordPath, dict_noteMapPath):
    note_name = np.array(['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'])
    n = 0
    dict_chordFromInt = {}
    dict_intFromChord = {}
    for filename in os.listdir(exampleDir):
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

        chroma = datawork.read_matrix(exampleDir + filename)
        datawork.save(chroma, refsDir + root + "/" + filename[0:len(filename) - 4] + ".pickle")
        dict_chordFromInt[n] = root + type
        dict_intFromChord[root + type] = n
        n += 1  # для словаря

        for i in range(1, len(note_name)):
            # Добавления пар (аккорд - цифра) в словари
            dict_chordFromInt[n] = note_name[i] + type
            dict_intFromChord[note_name[i] + type] = n

            # циклическая перестановка для создания хромаграммы каждого аккорда
            chroma = np.roll(chroma, 1, axis=0)
            datawork.save(chroma, refsDir + note_name[i] + "/" + note_name[i] + type + ".pickle")
            n += 1

    datawork.save(dict_chordFromInt, dict_chordFromIntPath)
    datawork.save(dict_intFromChord, dict_intFromChordPath)
    dict_noteMap = {'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb'}
    datawork.save(dict_noteMap, dict_noteMapPath)


# Преобразование аудио в нужный формат
def prepare_audio(audio_directory):
    for filename in os.listdir(audio_directory):
        if filename[len(filename) - 4:len(filename)] == ".mp3":
            name = filename[0:len(filename) - 4]
            mp3 = AudioSegment.from_mp3(audio_directory + filename)
            if mp3.channels > 1:
                mp3.set_channels(1)
            mp3.export(audio_directory + name + ".wav", format="wav")
            os.remove(audio_directory + filename)




