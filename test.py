from pydub import AudioSegment
import librosa
import datawork
import models
import numpy as np

audio = AudioSegment.from_wav("../songs_for_training/wav/Another Girl.wav")

start = 0
chromaSize = librosa.frames_to_time(5-1)
test = []
while (start < 30):
    test.append(datawork.get_chromagram_from_audio(audio, start, start+chromaSize))
    start += chromaSize
testArr = np.array(test)
# test_r = datawork.reduceAll(test.reshape(1, test.shape[0], test.shape[1]), 5)
test_denoised = models.denoise("models/denoise.h5", testArr)
result = test_denoised.reshape(test_denoised.shape[1], test_denoised.shape[2] * test_denoised.shape[0])
datawork.print_chromagram(result, "whole song")
