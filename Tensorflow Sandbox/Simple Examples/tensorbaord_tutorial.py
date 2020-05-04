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

now = datetime.utcnow().strftime("%Y_%m_%d_%H")
file_name = os.path.basename(__file__)
root_logdir = "tf_tensorboard/{}".format(file_name)
logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name ,now)

## accurancy function definition
def accurancy(y_pred, labels):
    """
    y_pred size = (N_samples x n_classes)
    labales -> true values, size = (N_samples x n_classes)
    """
    return np.sum(np.argmax(y_pred,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

## model builder
# input data is an 28x28 size image

batch_size = 100
layer_ids = ["hidden"+str(i+1) for i in range(5)]
layer_ids.append("out")
layer_sizes = [784, 500, 400, 300, 200, 100, 10]

#input definition
with tf.name_scope("training_data"):
    X_train = tf.placeholder(dtype=tf.float32, shape=[batch_size, layer_sizes[0]], name="training_input")
    y_train = tf.placeholder(dtype=tf.float32, shape=[batch_size, layer_sizes[-1]], name="training_labels")

#weights and bias definition
with tf.name_scope("weights_and_biases"):
    for idx, layer_name in enumerate(layer_ids):
            with tf.variable_scope(layer_name):
                weight = tf.get_variable(name="weight", shape=[layer_sizes[idx], layer_sizes[idx+1]], 
                                         initializer =tf.truncated_normal_initializer(stddev=0.05))
                bias = tf.get_variable(name="bias", shape=[layer_sizes[idx+1]], 
                                       initializer=tf.random_uniform_initializer(-0.1,0.1))
                
#model calculation definition
with tf.name_scope("model"):
    data = X_train
    for layer_name in layer_ids:
        with tf.variable_scope(layer_name, reuse=True):
            w, b = tf.get_variable("weight"), tf.get_variable("bias")
            if layer_name != "out":
                data = tf.nn.elu(tf.matmul(data, w)+b, name=layer_name+"_output")
            else:
                data = tf.nn.xw_plus_b(data, w, b, name=layer_name+"_output")
    predictions = tf.nn.softmax(data, name="predictions")

#trainig section definition
with tf.name_scope("trainig"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=data), name="cross_entropy_loss")
    learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    gradients = optimizer.compute_gradients(loss)
    training_op = optimizer.minimize(loss)
    
#summaries definition
with tf.name_scope("performance"):
    ##define sumries objects as placeholders -> tensorboard needs it
    #loss summary
    loss_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="loss_summary")
    loss_summary = tf.summary.scalar("Loss", loss_summary_ph) 
    #accurancy summary
    accuracy_summary_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
    accuracy_summary = tf.summary.scalar('accuracy', accuracy_summary_ph)
    
    # Gradient norm summary
    for g,v in gradients:
        if 'hidden5' in v.name and 'weight' in v.name:
            with tf.name_scope('gradients'):
                tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
                tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
                break
    
    #merging summaries
    merged_summaries = tf.summary.merge([loss_summary, accuracy_summary])

##Execution of the model

with tf.name_scope("initialization"):
    init = tf.global_variables_initializer()
    
#model params

image_size = 28
n_channels = 1
n_classes = 10
n_train = 55000
n_valid = 5000
n_test = 10000
n_epochs = 100

#configuration of gpu usage -> 0.9 means that gpu wont be overused
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9 

#session start
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

            if i == 0:
            # Only for the first epoch, get the summary data
            # Otherwise, it can clutter the visualization
                l,_, gn_summ = sess.run([loss, training_op, tf_gradnorm_summary],
                                        feed_dict={X_train: batch[0].reshape(batch_size,image_size*image_size),
                                                   y_train: batch[1],
                                                   learning_rate: 0.001})
                summary_writer.add_summary(gn_summ, epoch)
            
            else:
                l,_ = sess.run([loss, training_op],
                              feed_dict={X_train: batch[0].reshape(batch_size,image_size*image_size),
                                         y_train: batch[1],
                                         learning_rate: 0.001})
            loss_per_epoch.append(l)
        
        print('Average loss: {} in epoch: {}'.format(round(np.mean(loss_per_epoch),5), epoch))
        avg_loss = np.mean(loss_per_epoch)
        
        ## calculation validation accurancy
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

            
    

