import datawork

def getChords(filepath):
    chroma = datawork.get_chromagram(filepath).T
    chroma = chroma.reshape(chroma.shape[0], chroma.shape[1], 1)
    print()