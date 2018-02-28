import numpy as np
import matplotlib.pyplot as plt
import random


#Following code found online
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
 '''Return train_data, train_labels, test_data, test_labels
 The shape of data is 32 x 32 x3'''
 train_data = None
 train_labels = []

 for i in range(1, 6):
  data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
  if i == 1:
   train_data = data_dic[b'data']
  else:
   train_data = np.vstack((train_data, data_dic[b'data']))
  train_labels += data_dic[b'labels']

 test_data_dic = unpickle(data_dir + "/test_batch")
 test_data = test_data_dic[b'data']
 test_labels = test_data_dic[b'labels']

 train_data = train_data.reshape((len(train_data), 3, 32, 32))
 train_data = np.rollaxis(train_data, 1, 4)
 train_labels = np.array(train_labels)

 test_data = test_data.reshape((len(test_data), 3, 32, 32))
 test_data = np.rollaxis(test_data, 1, 4)
 test_labels = np.array(test_labels)

 return train_data, train_labels, test_data, test_labels

data_dir = 'cifar-10-batches-py'

train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)

print(train_data.shape)
print(train_labels.shape)

print(test_data.shape)
print(test_labels.shape)

#End of online code



#Converting Images To Grayscale
gsdata = []
for i in range(1000):
  image = []
  for j in range(32):
    image_row = []
    for k in range(32):
      temp = train_data[i,j,k].astype(int)
      gray = int((temp[0] + temp[1] + temp[2])/3)
      image_row.append(gray)
    image.append(image_row)
  gsdata.append(image)

#Testing
x = random.randint(0,1000)
plt.imshow(gsdata[x])
plt.show()

weights1 = np.random.randn(32,32)
hidden = np.zeros(1024)
output = np.zeros(100)