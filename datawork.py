import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import librosa
from librosa import display
import os
from sklearn.decomposition import PCA
from paths import Path
import pickle


# Нормализация матрицы
def normalize(matrix):
    min = matrix.min()
    max = matrix.max()
    if min != 0 and max != 0:
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
def get_noncategorical(categ):
    intToChord_dict = get(Path.Pickle.intToChord_dict)
    noncateg, confidence = [], []
    for array in categ:
        num = np.argmax(array)
        noncateg.append(intToChord_dict[num])
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

def get_beats(audioPath):
    y, sr = librosa.load(audioPath)
    tempo, beats_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats_time = []
    if beats_frames[0] <= 5:
        beats_frames = np.delete(beats_frames, 0)
    for i in range(1, len(beats_frames)):
        if not beats_frames[i] - beats_frames[i - 1] <= 5:
            beats_time.append(librosa.frames_to_time(beats_frames[i-1], sr=sr))
    beats_time.append(librosa.frames_to_time(beats_frames[len(beats_frames)-1], sr=sr))
    return np.array(beats_time)

