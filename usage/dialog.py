import tkinter as tk
from tkinter import filedialog
from usage import recognition

root = tk.Tk()
root.withdraw()

filepath = filedialog.askopenfilename()

recognition.getChords(filepath)
