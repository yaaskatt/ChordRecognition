import prep
import datawork
import train
import models

# Стадии выполнения задач

allData_read = True
chromaRefs_cr = True
audio_pr = True
denoiseModel_cr = True
classifierModel_cr = True

# Расположения файлов

allDataPath = "../songs_for_training/allData.pickle"

refsDir = "../chromagrams_for_training/chroma_refs/"
refsExampleDir = "../chromagrams_for_training/note_c_chroma_examples/"

dict_toChordPath = "dict/chord_from_int.pickle"
dict_toIntPath = "../dict/int_from_chord.pickle"
dict_noteMapPath = "..dict/note_map"

denoiserPath = "models/denoiser.h5"
classifierPath = "models/classifier.h5"
grouperPath = "models/grouper.h5"
reducerPath = "models/reducer.pickle"

songsPath = "../songs_for_training/songs_set.csv"
chordsDir = "../songs_for_training/chords/"
audioDir = "../songs_for_training/wav/"

n_reducerComp = 5

if not chromaRefs_cr:
    prep.create_references(refsDir, refsExampleDir, dict_toChordPath, dict_toIntPath, dict_noteMapPath)
if not audio_pr:
    prep.prepare_audio(audioDir)

# Получение ТРИАД (хромаграма - эталонная хромаграма в один столбец - аккорд) из датасета
if not allData_read:
    datawork.read_chords_data(songsPath, audioDir, chordsDir, refsDir, dict_noteMapPath, allDataPath)
chromas, chroma_refs, chords = datawork.get(allDataPath)

# Приведение ХРОМАГРАММ к одинаковому РАЗМЕРУ при помощи PCA
chromas_red = datawork.reduceAll(chromas, n_reducerComp)

# СОЗДАНИЕ модели автокодировщика (ДЕНОЙЗЕР)
if not denoiseModel_cr:
    train.train_denoiser_model(chromas_red, datawork.widenAll(chroma_refs, n_reducerComp), denoiserPath)

chromas_denoised = models.denoise(denoiserPath, chromas_red)

# Преобразование каждого АККОРДА в соответствующую ему ЦИФРУ (массив с одной единицей)
out_exp_categ = datawork.get_categorical(chords, dict_toIntPath)

# СОЗДАНИЕ модели сверточной сети (КЛАССИФИКАТОР)
if not classifierModel_cr:
    train.train_classifier_model(chromas_denoised, out_exp_categ, classifierPath)

# Выход классификатора в виде МАССИВА
out_categ = models.classify(classifierPath, chromas_denoised)

# Выход классификатора в виде НАЗВАНИЯ АККОРДА
out_noncateg, confidence = datawork.get_noncategorical(out_categ, dict_toChordPath)

datawork.print_chromagram(chromas[0])
datawork.print_chromagram(chromas_red[0])
datawork.print_chromagram(chromas_denoised[0])

for i in range(len(chords)):
    print("Ожидаемый аккорд: ", chords[i],
          "    Полученный аккорд: ", out_noncateg[i],
          "    Уверенность: ", confidence[i], sep="")


