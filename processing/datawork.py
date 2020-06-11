import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
import librosa
from librosa import display
import os
from sklearn.decomposition import PCA
from processing.paths import Path
import pickle
from pydub import AudioSegment

# Нормализация матрицы
def normalize(matrix):
    min = matrix.min()
    max = matrix.max()
    if min != 0 and max != 0:
        for i in range(len(matrix)):
            matrix[i] = (matrix[i] - min) / (max - min)
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

def print_chromagram(chromagram):
    plt.figure(figsize=(20, 5))
    display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')
    plt.show()

# Уменьшение размерности хромаграмм до нужного значения
def reduceAll(x, n_components):
    x_red = []
    for chroma in x:
        x_red.append(reduce(chroma, n_components))
    return np.array(x_red)

# Уменьшение размерности матрицы
def reduce(chroma, n_components):
    pca = PCA(n_components=n_components)
    chroma = normalize(chroma)
    chroma = pca.fit_transform(chroma)
    return chroma

# Преобразовать массив аккордов в категориальный формат
def get_categorical(chords):
    noncateg = get_int(chords)
    categ = to_categorical(noncateg)
    return categ

def get_categorical_from_int(noncateg):
    categ = to_categorical(noncateg)
    return categ

def get_int(chords):
    chord_to_int = get(Path.Pickle.chordToInt_dict)
    int_chords = []
    for i in range(len(chords)):
        int_chords.append(chord_to_int[chords[i]])
    return int_chords

def get_int_array(chords):
    int_chords = []
    for i in range(len(chords)):
        int_chords.append(get_int(chords[i]))
    return int_chords

def get_chordNames(int_chords):
    intToChord_dict = get(Path.Pickle.intToChord_dict)
    chordNames = []
    for j in range(len(int_chords)):
        chordNames.append(intToChord_dict[int_chords[j]])
    return np.array(chordNames)

# Преобразовать категорический массив в желаемый вид
def get_noncategorical(categ):
    intToChord_dict = get(Path.Pickle.intToChord_dict)
    noncateg, confidence = [], []
    array = categ.tolist()
    for j in range(len(array)):
        num = np.argmax(array[j])
        noncateg.append(intToChord_dict[num])
        confidence.append(max(array[j]))
    return noncateg, confidence

def get_beats(audioPath):
    y, sr = librosa.load(audioPath)
    tempo, beats_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = []
    if beats_frames[0] <= 5:
        beats_frames = np.delete(beats_frames, 0)
    for i in range(1, len(beats_frames)):
        if not beats_frames[i] - beats_frames[i - 1] <= 5:
            beats.append(beats_frames[i-1])
    beats.append(beats_frames[len(beats_frames)-1])
    return np.array(beats)

# Преобразование фреймы в секунды, используя номера битов
def get_time(beats, beat_nums):
    beat_nums = [int(i) for i in beat_nums]
    time = []
    for i in range(len(beat_nums)):
        time.append(librosa.frames_to_time(beats[beat_nums[i]]))
    return np.array(time)

def mp3_to_wav(audioPath):
    filename = os.path.basename(audioPath)
    if filename[len(filename) - 4:len(filename)] == ".mp3":
        name = filename[0:len(filename) - 4]
        mp3 = AudioSegment.from_mp3(audioPath)
        if mp3.channels > 1:
            mp3 = mp3.set_channels(1)
        mp3.export(os.path.dirname(audioPath) + name + ".wav", format="wav")
    return os.path.dirname(audioPath) + name + ".wav"