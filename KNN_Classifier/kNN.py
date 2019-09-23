import numpy as np
import math

def original_max(sequence):
    maximum = sequence[0]
    for item in sequence:
        if item[1] > maximum[1]:
            maximum = item

    list_max = []
    for item in sequence:
        if item[1] == maximum[1]:
            list_max.append(item)
    return list_max

def kNN(train, test, a_num, dat_num,class_label, k, weighted):
    test_d = np.tile(test, (dat_num, 1))
    differece = train - test_d
    distance = np.full((dat_num, 1), 0, dtype="float64")
    for i in range(0, dat_num):
        sum = 0
        for item in differece[i, :]:
            sum += item*item
        distance[i] = sum

    distance = distance.tolist()
    class_label = class_label.tolist()

    for i in range(0, len(distance)):
       distance[i].append(class_label[i])

    distance.sort()
    oracle = np.array(distance)[0:k, :]
    unique= np.unique(oracle[:, 1])
    unique = unique.tolist()

    stats = []

    for item in unique:
        stats.append([item, 0])

    if not weighted:
        for i in range(0, len(oracle)):
            for stat in stats:
                if oracle[i, 1] in stat:
                    stat[1] += 1

    if weighted:
        for i in range(0, len(oracle)):
            for stat in stats:
                if oracle[i, 1] in stat:
                    stat[1] += 1/(float(oracle[i,0])**2)

    max = original_max(stats)
    for item in max:
        print(item[0])

