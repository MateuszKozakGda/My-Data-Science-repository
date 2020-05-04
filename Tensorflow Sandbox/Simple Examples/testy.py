import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

batch_size = 50

housing = fetch_california_housing()
m, n = housing.data.shape

scaller = StandardScaler()
scalled_data = scaller.fit_transform(housing.data)

#number of batches
batches = int(np.ceil(m / batch_size))

#print(m,n)
scalled_housing_data_plus_bias = np.c_[np.ones((m,1)), scalled_data]

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * batches + batch_index)  # nieukazane w książce
    indices = np.random.randint(m, size=batch_size)  # nieukazane
    X_batch = scalled_housing_data_plus_bias[indices] # nieukazane
    y_batch = housing.target.reshape(-1, 1)[indices] # nieukazane
    return X_batch, y_batch

for i in range(5):
    for x in range(batches):
        X, y = fetch_batch(i, x, batch_size)
        #X = X.astype("float32")
        print(f"X : \n{X.dtype}")
        
        
        