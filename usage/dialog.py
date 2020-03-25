import tkinter as tk
from tkinter import filedialog
from usage import recognition
from processing import datawork

root = tk.Tk()
root.withdraw()

filepath = filedialog.askopenfilename()
filepath = datawork.mp3_to_wav(filepath)
recognition.getChords(filepath)
