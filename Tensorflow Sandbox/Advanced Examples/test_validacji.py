from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
X_train = mnist.train.images
y_train = mnist.train.labels

import numpy as np

def _get_validation_set(X, y, validation_split=0.2):
    
    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(validation_split*len(value_inds))

        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True
    X_train, X_val = X[train_inds], X[test_inds]
    y_train, y_val = y[train_inds], y[test_inds]

    return X_train, X_val, y_train, y_val
    
a,b,c,d = _get_validation_set(X_train, y_train)


from sklearn.model_selection import train_test_split

a,b,c,d = train_test_split(X_train,y_train, test_size=0.2, random_state=2)

print(f"a shape: {a.shape}, b shape: {b.shape}, c shape: {c.shape}, d shape: {d.shape}")