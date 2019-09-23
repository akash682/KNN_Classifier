import numpy as np
import preprocess_data as p
import readfile as r
import kNN

data_train = r.readfile("Text_Data_for_Project1_train_data.txt")
data_test = r.readfile("Text_Data_for_Project1_test_data.txt")

alabel_train, clabel_train, adata_train, cdata_train = p.p_traindata(data_train)
alabel_test, adata_test = p.p_testdata(data_test)

atraintestmix = np.vstack((adata_train, adata_test))

att_num = np.shape(adata_train)[1]
dat_num = np.shape(adata_train)[0]

adata_traintest_dummy, a_label = p.indexing(atraintestmix, att_num, dat_num)
adata_train_dummy = adata_traintest_dummy[:-1, :]
adata_test_dummy = np.array([adata_traintest_dummy[-1, :]])
cdata_train_dummy, c_label = p.indexing(cdata_train, 1, dat_num)

k = 9
weighted = 1
kNN.kNN(adata_train_dummy, adata_test_dummy, len(a_label), dat_num, cdata_train, k, weighted)
