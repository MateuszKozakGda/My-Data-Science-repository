import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
#print(m,n)
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]

##graph definition
#using normal equation: theta = ((XT * X)^-1)*XT*y
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype = tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)), XT), y)

with tf.compat.v1.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)