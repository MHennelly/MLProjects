import numpy as np
import matplotlib.pyplot as mpl

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

db1 = unpickle("cifar-10-batches-py/data_batch_1")
db2 = unpickle("cifar-10-batches-py/data_batch_2")
db3 = unpickle("cifar-10-batches-py/data_batch_3")
db4 = unpickle("cifar-10-batches-py/data_batch_4")
db5 = unpickle("cifar-10-batches-py/data_batch_5")


#Visualize
mpl.imshow(db1)