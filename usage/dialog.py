import tkinter as tk
from tkinter import filedialog
from usage import recognition
from processing import datawork
from tkinter.ttk import *

class Out(Frame):

    def __init__(self, parent):
        filepath = filedialog.askopenfilename()
        filepath = datawork.mp3_to_wav(filepath)
        label = Label(text='Подождите')

        Frame.__init__(self, parent)
        chords = recognition.getChords(filepath)
        self.CreateUI()
        self.LoadTable(chords)
        self.grid(sticky="nswe")
        parent.grid_rowconfigure(0, weight = 1)
        parent.grid_columnconfigure(0, weight = 1)

    def CreateUI(self):
        tv = Treeview(self)

        tv['show'] = 'headings'
        tv['columns'] = ('start', 'end', 'chord')
        tv.heading('start', text='Start Time')
        tv.column('start', anchor='center', width=100)
        tv.heading('end', text='End Time')
        tv.column('end', anchor='center', width=100)
        tv.heading('chord', text='Chord')
        tv.column('chord', anchor='center', width=100)
        tv.grid(sticky="nswe")
        self.tree = tv

        self.yscrollbar = Scrollbar(self, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.yscrollbar.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.yscrollbar.grid(row=0, column=1, sticky='nse')
        self.yscrollbar.configure(command=self.tree.yview)
        self.grid_rowconfigure(0, weight = 1)
        self.grid_columnconfigure(0, weight = 1)

    def LoadTable(self, array):
        for i in range(len(array)):
            self.tree.insert('', 'end', values=(array[i][0],
                             array[i][1], array[i][2]))

root = tk.Tk()
Out(root)
root.mainloop()



