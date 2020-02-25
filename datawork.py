import pandas as pd
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import librosa
from librosa import display
import pickle
import os
from sklearn.decomposition import PCA


# Нормализация матрицы
def normalize(matrix):
    min = matrix.min()
    max = matrix.max()
    for i in range(len(matrix)):
        matrix[i] = (matrix[i] - min) / (max - min)
    return matrix


# Чтение матрицы из файла
def read_matrix(filename):
    f = open(filename)
    matrix = f.readline().replace("\n", "").split(" ")
    for line in f:
        new_row = line.replace("\n", "").split(" ")
        matrix = np.vstack([matrix, new_row])
    return matrix


# Увеличение размерности эталонных хромаграмм до размерности обрабатываемых
def widenAll(chromaRefs, columns_new):
    chromas_wide = []
    for chroma in chromaRefs:
        chromas_wide.append(widen(chroma, columns_new))
    return np.array(chromas_wide)


def widen(array, columns_new):
    if columns_new == 2:
        return np.column_stack((array, array))
    return np.column_stack((array, widen(array, columns_new - 1)))


def get_chromagram(filePath):
    y, sr = librosa.load(filePath)
    chromagram = librosa.feature.chroma_stft(y, sr=sr)
    return chromagram


def print_chromagram(chromagram):
    plt.figure(figsize=(20, 5))
    display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.show()


# Сохранение объекта при помощи pickle
def save(obj, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs((os.path.dirname(path)))

    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# Открытие объекта при помощи pickle
def get(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Уменьшение размерности хромаграмм до нужного значения
def reduceAll(x, n_components):
    pca = PCA(n_components=n_components)
    x_red = []
    for chroma in x:
        x_red.append(reduce(pca, chroma))
    return np.array(x_red)


def reduce(pca, chroma):
    chroma = normalize(chroma)
    chroma = pca.fit_transform(chroma)
    return chroma


# Преобразовать массив аккордов в категорический массив
def get_categorical(chords, int_from_chords_dict_path):
    int_from_chords = get(int_from_chords_dict_path)
    noncateg = []
    for chord in chords:
        noncateg.append(int_from_chords[chord])
    categ = to_categorical(noncateg)
    return categ


# Преобразовать категорический массив в желаемый вид
def get_noncategorical(categ, chords_from_int_dict_path):
    int_to_chords = get(chords_from_int_dict_path)
    noncateg, confidence = [], []
    for array in categ:
        num = np.argmax(array)
        noncateg.append(int_to_chords[num])
        confidence.append(max(array))
    return noncateg, confidence


# Получение хромаграммы указанного отрывка
def get_chromagram_from_audio(audio, start, end):
    temp_segment_path = "tempSegment.wav"
    audio_segment = audio[float(start) * 1000:float(end) * 1000]
    audio_segment.export(temp_segment_path, format="wav")
    chromagram = get_chromagram(temp_segment_path)
    os.remove(temp_segment_path)
    return chromagram

def get_bpm(audioPath):
    y, sr = librosa.load(audioPath)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
    print(dtempo)
    return dtempo


# Чтение данных из датасетов
def read_data_to_chroma_set(songsSet, audioDir, chordsDir, refsDir, dict_noteMapPath, allDataPath):

    dict_noteMap = get(dict_noteMapPath)
    songs = pd.read_csv(songsSet, sep=";", encoding="UTF-8")
    chromas, chromaRefs, chords = [], [], []

    for i in range(songs.shape[0]):
        audiofile, chords_file = songs.iloc[i]
        chordSet = pd.read_csv(chordsDir + chords_file, sep=";", encoding="UTF-8", keep_default_na=False)
        audio = AudioSegment.from_wav(audioDir + audiofile)

        for j in range(chordSet.shape[0]):
            start, end, root, type = chordSet.iloc[j]

            if j == chordSet.shape[0] - 1:
                end = audio.duration_seconds
            if root == "N":
                continue
            # Замена нот с диезом на ноты с бемолем
            if root in dict_noteMap:
                root = dict_noteMap[root]

            print(audiofile)
            chromas.append(get_chromagram_from_audio(audio, start, end))
            chromaRefs.append(get(refsDir + root + "/" + root + type + ".pickle"))
            chords.append(root + type)
    save((chromas, np.array(chromaRefs), np.array(chords)), allDataPath)

