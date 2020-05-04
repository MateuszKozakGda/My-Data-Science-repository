import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from reset_graph import reset_graph
##hiding errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##saving directories
from datetime import datetime

now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
file_name = os.path.basename(__file__)
root_logdir = "tf_tensorboard/{}".format(file_name)
logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name ,now)

##reset graph
reset_graph()

## accurancy function definition
def accurancy(y_pred, labels):
    """
    y_pred size = (N_samples x n_classes)
    labales -> true values, size = (N_samples x n_classes)
    """
    return np.sum(np.argmax(y_pred,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

def model(X_train, config, training, dropout_rate =0.5, momentum=0.99):
    """
    X -> input data features
    Config --> list with number of layers, config[0] = X.shape[0], config[-1] = number of outputs
    training --> default placeholder, True is for training, false for working
    """
    with tf.name_scope("model"):
        
        weight_init = tf.contrib.layers.variance_scaling_initializer()
        bias_init = tf.random_uniform_initializer(-0.1,0.1)
        
        for index, layer_size in enumerate(config):
            with tf.name_scope("layer"+str(index+1)):
                if index == 0: 
                    hidden = tf.layers.dense(X_train, layer_size, name="input",
                                            kernel_initializer=weight_init, use_bias=True, bias_initializer=bias_init)
                    bn_hidden = tf.layers.batch_normalization(hidden,training=training, momentum=momentum)
                    activation = tf.nn.elu(bn_hidden)
                    drop = tf.layers.dropout(activation, dropout_rate, training=training)
                elif index != 0 or index != len(config)-1:
                    hidden = tf.layers.dense(drop, layer_size, name="hidden"+str(index),
                                            kernel_initializer=weight_init, use_bias=True, bias_initializer=bias_init)
                    bn_hidden = tf.layers.batch_normalization(hidden,training=training, momentum=momentum)
                    activation = tf.nn.elu(bn_hidden)
                    drop = tf.layers.dropout(activation, dropout_rate, training=training)
                else:
                    hidden = tf.layers.dense(drop, layer_size, name="output",
                                            kernel_initializer=weight_init, use_bias=True, bias_initializer=bias_init)
                    bn_hidden = tf.layers.batch_normalization(hidden,training=training, momentum=momentum)
                    activation = tf.nn.elu(bn_hidden)
                    
                
    return activation

##cdefining model params
image_size = 28
n_channels = 1
n_classes = 10
n_train = 55000
n_valid = 5000
n_test = 10000
n_epochs = 10

layer_sizes = [image_size*image_size, 500, 400, 300, 200, 100, n_classes] 
batch_size = 300

##defining place holders

X_train = tf.placeholder(dtype=tf.float32, shape=[batch_size, layer_sizes[0]], name="X")
y_train = tf.placeholder(dtype=tf.float32, shape=[batch_size, layer_sizes[-1]], name="y")

training = tf.placeholder_with_default(False, shape=(), name="is_training")

logits = model(X_train, config=layer_sizes, training=training, dropout_rate=0.3, momentum=0.99)
predictions = tf.nn.softmax(logits, name="predictions")

with tf.name_scope("training"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=logits, name="cross_entropy_loss"))
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(loss)
    training_op = optimizer.minimize(loss)

with tf.name_scope("performance"):
    ##define sumries objects as placeholders -> tensorboard needs it
    #loss summary
    loss_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="loss_summary")
    loss_summary = tf.summary.scalar("Loss", loss_summary_ph) 
    #accurancy summary
    accuracy_summary_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    accuracy_summary = tf.summary.scalar('accuracy', accuracy_summary_ph)
    merged_summaries = tf.summary.merge([loss_summary, accuracy_summary])

with tf.name_scope("initialization"):
    init = tf.global_variables_initializer()

with tf.name_scope("extra_operations"):  
    extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) ## for moving average update for batch nowmalization layer

# extra config
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 

with tf.compat.v1.Session(config=config) as sess:
    
    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    #initialize all variables
    sess.run(init)

    accuracy_per_epoch = []
    mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

    for epoch in range(n_epochs):
        loss_per_epoch = []
        print("===========EPOCH: {} ===========".format(epoch))
        
        for i in range(n_train//batch_size):
            batch = mnist_data.train.next_batch(batch_size)
            
            loop_count = round(n_train//batch_size,-1)
            loop_divider = round(loop_count//20,-1)
            
            # Only for the first epoch, get the summary data
            # Otherwise, it can clutter the visualization
            l,_, ex_op = sess.run([loss, training_op, extra_ops],
                                    feed_dict={X_train: batch[0].reshape(batch_size,image_size*image_size),
                                                y_train: batch[1],
                                                training: True,
                                                learning_rate: 0.02})
            loss_per_epoch.append(l)
        print('Average loss: {} in epoch: {}'.format(round(np.mean(loss_per_epoch),5), epoch))
        avg_loss = np.mean(loss_per_epoch)
            
        valid_accuracy_per_epoch = []
        for i in range(n_train//batch_size):
            valid_images,valid_labels = mnist_data.validation.next_batch(batch_size)
            valid_batch_predictions = sess.run(
                predictions, feed_dict={X_train: valid_images.reshape(batch_size,image_size*image_size)})
            valid_accuracy_per_epoch.append(accurancy(valid_batch_predictions,valid_labels))

        mean_v_acc = np.mean(valid_accuracy_per_epoch)
        print('Average Valid accurancy: {} in epoch: {}'.format(round(np.mean(valid_accuracy_per_epoch),5), epoch))
        
        #perfomance savig to tensorbaord
        accuracy_per_epoch = []
        for i in range(n_test//batch_size):
            test_images, test_labels = mnist_data.test.next_batch(batch_size)
            test_batch_predictions = sess.run(
                predictions,feed_dict={X_train: test_images.reshape(batch_size,image_size*image_size)}
            )
            accuracy_per_epoch.append(accurancy(test_batch_predictions,test_labels))

        print('Average test accurancy: {} in epoch: {}'.format(round(np.mean(accuracy_per_epoch),5), epoch))
        avg_test_accuracy = np.mean(accuracy_per_epoch)

        # Execute the summaries
        summary = sess.run(merged_summaries, feed_dict={loss_summary_ph:avg_loss, accuracy_summary_ph:avg_test_accuracy})

        # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
        summary_writer.add_summary(summary, epoch)
        print("<=========== END ===========>")  
    