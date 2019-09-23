import numpy as np

def readfile(fname):
    file = open(fname, "r")
    i = 0

    while True:
        line = file.readline()
        if not line:
            break
        tmpline = " ".join(line.split())
        temp = np.array([tmpline.split(" ")])
        if i == 0:
            data = temp
            i += 1
        else:
            try:
                data = np.concatenate((data, temp), axis=0)
            except:
                continue

    file.close()
    return data