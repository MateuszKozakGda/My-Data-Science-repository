import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

#
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##epoch definition and learning rate
n_epochs = 10000
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

scaller = StandardScaler()
scalled_data = scaller.fit_transform(housing.data)

#print(m,n)
scalled_housing_data_plus_bias = np.c_[np.ones((m,1)), scalled_data]

##graph definition
#using SGD hand written -> y_pred = (X*theta)+bias
X = tf.constant(scalled_housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype = tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0,1), name="theta") #inicjalizacja w przedziale -1,1
y_pred = tf.matmul(X,theta, name="prognozy")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse") ##obliczaie bledu kwadratowego
gradients = 2/m * tf.matmul(tf.transpose(X), error) ## SGD wyliczanie graduentu prostego
training_op = tf.assign(theta, theta-learning_rate*gradients) ## zmiana wartosci parametru theta o wyliczony gradient

##inicjalizacja zmiennych w wezlach
tf.set_random_seed(5)
init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init) ## inicjalizacja w sesji
    
    for epochs in range(n_epochs):
        if epochs % 100 == 0:
            print(f"Epoka: {epochs}, MSE: {mse.eval()}")
            sess.run(training_op) ## uczenie ukladu, zmiana gradientu
        
    best_theta = theta.eval()
print(f"Theta ostateczne : {best_theta}")
