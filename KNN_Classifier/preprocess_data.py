import numpy as np

def p_traindata(data):
    alabel = data[0, :-1]
    clabel = data[0, -1]
    adata = data[1:, 1:-1]
    cdata = data[1:, -1]
    return alabel, clabel, adata, cdata

def p_testdata(data):
    alabel = data[0, :]
    adata = data[1:, 1:]
    return alabel, adata

def indexing(data_train, att_num, dat_num):
    empty_bigseeds = 1
    big_seeds = np.array([])
    label_list = []
    for i in range(0, att_num):
        try:
            data_att = data_train[:, i]
        except:
            data_att = data_train
        dlabel = np.array(list(set(data_att)))
        label_list.extend(dlabel)
        empty_seeds = 1
        seeds = np.array([])
        for label in dlabel:
            boolArr = np.where(data_att == label, 1, 0)
            if empty_seeds == 1:
                seeds = boolArr
                empty_seeds = 0
            else:
                seeds = np.vstack([seeds, boolArr])

        seeds = np.transpose(seeds)
        if empty_bigseeds == 1:
            big_seeds = seeds
            empty_bigseeds = 0
        else:
            big_seeds = np.hstack((big_seeds,seeds))
    return big_seeds, label_list