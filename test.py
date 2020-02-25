from pydub import AudioSegment
import librosa
import datawork


y, sr = librosa.load("../songs_for_training/wav/Another Girl.wav")
print(y, sr)
y, index = librosa.effects.trim(y)
print(y, sr)
librosa.output.write_wav("audioFile.wav", y, sr)
bpm = datawork.get_bpm("audioFile.wav")
audio = AudioSegment.from_wav("audioFile.wav")


start = 0
end = 60/bpm[5] * 10
print(end)
segment1 = audio[start :end * 1000]
segment1.export("segment1.wav", format="wav")