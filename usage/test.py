from usage import recognition
from processing.paths import Dir
from keras.models import load_model
from processing.paths import Path
from memory_profiler import profile
from processing import datawork

recognition.getChords(Dir.audioSet + '/' + "All My Loving.wav")
