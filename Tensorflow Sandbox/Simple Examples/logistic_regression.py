import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from modulowosc1 import reset_graph
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score

reset_graph()

## hiding warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##data for file managment
from datetime import datetime

now = datetime.utcnow().strftime("%Y_%m_%d_%H:%M:%S")
file_name = os.path.basename(__file__)
root_logdir = "tf_dzienniki/{}".format(file_name)
logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name ,now)

## generate data for regression
samples = 1000
X_moon, y_moon = make_moons(samples, noise=0.12, random_state=42)

##showing data
plt.plot(X_moon[y_moon == 1, 0], X_moon[y_moon == 1, 1], 'go', label="Pozytywna")
plt.plot(X_moon[y_moon == 0, 0], X_moon[y_moon == 0, 1], 'r^', label="Negatywna")
plt.legend()
#plt.show()

print(X_moon.shape)

##data transformations with polynominal features
data_lengh = X_moon.shape[0]

poly = PolynomialFeatures(3)
X_moon_transformed = poly.fit_transform(X_moon)
y_moon = y_moon.reshape(-1,1)

##splitting data to train and test samples

X_train, X_test, y_train, y_test = train_test_split(X_moon_transformed, y_moon, test_size=0.33, random_state=42)

## definig logstic regression model
# model params
n_epochs = 1000
learning_rate = 0.01
batch_size = 50
n_batches = int(np.ceil(X_moon_transformed.shape[0]/batch_size))

def get_random_batch(X,y, batch_size):
    random_index = np.random.randint(0, len(X), batch_size)
    X_batch = X[random_index]
    y_batch = y[random_index]
    return X_batch, y_batch

def cross_entropy(y, y_proba, epsilon = 1e-7):
    loss = -1*tf.reduce_mean(y*tf.log(y_proba+epsilon)+(1-y)*tf.log(1-y_proba+epsilon)) ##cross entropy formula
    return loss

def sigmoid(y):
    return 1/(1+tf.exp(-y)) 


#model variables
X = tf.placeholder(dtype = tf.float32, shape=(None, X_moon_transformed.shape[1]), name="input_data")
y = tf.placeholder(dtype = tf.float32, shape=(None, 1), name="results")
theta = tf.Variable(tf.random_uniform([X_moon_transformed.shape[1],1], -1.0, 1.0, seed=42), name="theta") ## size of X n_columns

y_pred = tf.matmul(X, theta, name="logits")
y_proba = sigmoid(y_pred)
loss = cross_entropy(y, y_proba)
#loss = tf.losses.log_loss(y,y_proba)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_inddex in range(n_batches):
            X_batch, y_batch = get_random_batch(X_train, y_train, batch_size=batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        loss_val = loss.eval({X: X_test, y: y_test})
        
        if epoch % 100 == 0:
            print("Epoka:", epoch, "\tFunkcja straty:", loss_val)
            
    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    y_prediction = [1 if y_proba_val[i] >= 0.5 else 0 for i in range(len(y_proba_val))]
    
    score_precision = precision_score(y_test, y_prediction)
    score_recall = recall_score(y_test, y_prediction)
    print("Precison: {}, Recall: {}".format(round(score_precision,6), round(score_recall,6)))
    


    

 


