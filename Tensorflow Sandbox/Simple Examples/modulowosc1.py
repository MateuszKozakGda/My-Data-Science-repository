import tensorflow as tf
import os
import numpy as np
from housing_3_minibatch_saver_tensorboard import reset_graph

## hiding warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##data for file managment
from datetime import datetime

now = datetime.utcnow().strftime("%Y_%m_%d_%H:%M:%S")
file_name = os.path.basename(__file__)
root_logdir = "tf_dzienniki/{}".format(file_name)
logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name ,now)

##def linear activation unit
## reseting graph
reset_graph()
tf.compat.v1.reset_default_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]),1)
        w = tf.Variable(tf.random_normal(w_shape), name="wagi")
        b = tf.Variable(0.0, name="obciazenie")
        z = tf.add(tf.matmul(X,w), b, name="result")
        return tf.maximum(z, 0., name="relu")

##
n_features = 5
X = tf.placeholder(dtype = tf.float32, shape=(None, n_features), name = "X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="wyjscie")

##saving graph
file_writer = tf.summary.FileWriter(root_logdir, tf.get_default_graph())