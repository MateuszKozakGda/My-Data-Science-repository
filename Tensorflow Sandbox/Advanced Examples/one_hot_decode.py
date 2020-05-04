import numpy as np

n_stages = 50

data = np.array([int(np.random.uniform(0,10,1)) for x in range(n_stages)])
#print(data)

shape = (n_stages, data.max()+1)
one_hot = np.zeros(shape)
rows = np.arange(n_stages)
one_hot[rows, data] = 1
#print(one_hot)

def one_hot_encode(input, n_classes):
    shape = (input.shape[0], n_classes)
    one_hot = np.zeros(shape)
    rows = np.arange(input.shape[0])
    one_hot[rows, input] = 1
    return one_hot
        
#print(one_hot_encode(data, 10))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


##taking 0-4 digits
#X_train1 = mnist.train.images[mnist.train.labels < 5]
#y_train1 = mnist.train.labels[np.argmax(mnist.train.labels) < 5]

"""zeros = np.zeros(mnist.train.labels.shape[1])
lista = []
for i in range(5):
    zeros = np.zeros(mnist.train.labels.shape[1], int)
    zeros[i] = 1
    lista.append(zeros)
print(lista)"""

def get_indices(y):
    lista_index = []
    for ind in range(y.shape[0]):
        if y[ind] < 5:
            lista_index.append(ind)
    return lista_index

index = get_indices(mnist.train.labels)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
y_train1 = mnist.train.labels[index]