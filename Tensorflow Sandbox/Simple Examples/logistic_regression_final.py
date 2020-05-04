import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from reset_graph import reset_graph

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


def my_logistic_regression(X, y, seed=42, learning_rate=0.01, momentum=0.9):
    n_inputs = int(X.get_shape()[1])
    with tf.name_scope("regresja_logistyczna"):
        with tf.name_scope("model"):
            initializer = tf.random_uniform([n_inputs,1], -1.0, 1, seed=seed)
            theta = tf.Variable(initializer, name="teta")
            logits = tf.matmul(X, theta, name="logity")
            y_proba = sigmoid(logits)
            
        with tf.name_scope("uczenie"):
            loss = cross_entropy(y, y_proba)
            if momentum:
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
            else:
                 optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar("log_loss", loss)
            
        with tf.name_scope("inicjalizacja_zmiennych"):
            init = tf.global_variables_initializer()
            
        with tf.name_scope("zapisywanie"):
            saver = tf.train.Saver()
        
    return y_proba, loss, training_op, loss_summary, init, saver

## definicja placeholderow
X = tf.placeholder(dtype = tf.float32, shape=(None, X_moon_transformed.shape[1]), name="input_data")
y = tf.placeholder(dtype = tf.float32, shape=(None, 1), name="results")

y_proba, loss, training_op, loss_summary, init, saver = my_logistic_regression(X, y, learning_rate=0.01)           

## saving paths
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

checkpoint_path = "{}/{}_przebieg-{}/logistic_regression.ckpt".format(root_logdir, file_name ,now)
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "{}/{}_przebieg-{}/final_model/logistic_regression_model".format(root_logdir, file_name ,now)


with tf.compat.v1.Session() as sess:
    ##restoring model or run new session
    if os.path.isfile(checkpoint_epoch_path):
        ##wczytywanie od ostatniej zapisanej epoki
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Uczenie zostało przerwane. Wznawiam od epoki".foramt(start_epoch))
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)
    
    ##training model
    for epoch in range(start_epoch, n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = get_random_batch(X_train, y_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
        loss_val, loss_str = sess.run([loss, loss_summary], feed_dict={X: X_test, y: y_test})
        
        y_preds_eval = y_proba.eval(feed_dict={X: X_test, y: y_test})
        y_prediction = [1 if y_preds_eval[i] >= 0.5 else 0 for i in range(len(y_preds_eval))]
        
        prec_str = precision_score(y_test, y_prediction)
        recal_str = recall_score(y_test, y_prediction)

        file_writer.add_summary(loss_str, epoch)

        if epoch % 100 == 0:
            print("Epoka: {}, strata: {}\n prec: {}, recall: {} ".format(epoch, round(loss_val, 6), round(prec_str, 6), round(recal_str, 6)))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
        
    saver.save(sess, final_model_path)
    y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
    os.remove(checkpoint_epoch_path)
    
            
    