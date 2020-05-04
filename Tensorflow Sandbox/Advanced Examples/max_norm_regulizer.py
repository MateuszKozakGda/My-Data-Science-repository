import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

##hiding errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##saving directories
from datetime import datetime

now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
file_name = os.path.basename(__file__)
root_logdir = "tf_tensorboard_minist/{}".format(file_name)
logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name ,now)
checkpoint_path = "{}/{}_przebieg-{}/Minist_model.ckpt".format(root_logdir, file_name ,now)

##reset graph

def reset_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()
tf.reset_default_graph()

weight_init = tf.contrib.layers.variance_scaling_initializer()
bias_init = tf.random_uniform_initializer(-0.1,0.1)

def max_norm_regularizer(threshold=1, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights) ## ---> trzeba wywolac funckcja get_collection()!!!
        return None 
    return max_norm

def accurancy(y_pred, labels):
    """
    y_pred size = (N_samples x n_classes)
    labales -> true values, size = (N_samples x n_classes)
    """
    return np.sum(np.argmax(y_pred,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

def model(X_train, y_train, n_hidden_layers, layer_size, training,
           activation_func = tf.nn.elu, batch_norm=True, dropout=True,dropout_rate =0.5, momentum=0.99):
    """
    X -> input data features
    n_hidden_layers --> number of hidden dense layers
    layer_size --> number of neurons in each layer
    training --> default placeholder, True is for training, false for working
    kernel_reg --> kernel regulaizer, can be used any tf/keras regulizer
    activation_func --> activation function, can be used every tf/keras or custom function
    batch_norm --> True if you want to use batch_norm every layer
    dropout --> true if you want to use droput function
    dropout_rate --> drop probability
    momentum --> batch norm momentum value used for updating moving averages
    """
    with tf.name_scope("model"):
        
        for i in range(n_hidden_layers):
            with tf.name_scope("layer"+str(i+1)):
                if i == 0: 
                    hidden = tf.layers.dense(X_train, layer_size, name="input",
                                            kernel_initializer=weight_init, use_bias=True, bias_initializer=bias_init, 
                                            kernel_regularizer=max_norm_regularizer(threshold=1.0))
                    if batch_norm:
                        hidden = tf.layers.batch_normalization(hidden,training=training, momentum=momentum)
                    hidden = activation_func(hidden)
                    if dropout:
                        hidden = tf.layers.dropout(hidden, dropout_rate, training=training)
                    
                else:
                    hidden = tf.layers.dense(hidden, layer_size, name="hidden"+str(i+1),
                                            kernel_initializer=weight_init, use_bias=True, bias_initializer=bias_init, 
                                            kernel_regularizer=max_norm_regularizer(threshold=1.0))
                    if batch_norm:
                        hidden = tf.layers.batch_normalization(hidden,training=training, momentum=momentum)
                    hidden = activation_func(hidden)
                    if dropout:
                        hidden = tf.layers.dropout(hidden, dropout_rate, training=training)

        #output layer:
        with tf.name_scope("Output"):    
            hidden = tf.layers.dense(hidden, y_train.shape[1], name="output_layer",
                                    kernel_initializer=weight_init, use_bias=True, bias_initializer=bias_init, 
                                    kernel_regularizer=max_norm_regularizer(threshold=1.0))
            if batch_norm:
                hidden = tf.layers.batch_normalization(hidden, training=training, momentum=momentum)
            hidden = activation_func(hidden, name="Output")    
         
    return hidden

## model params
if __name__ == '__main__':
    reset_graph()

    image_size = 28
    n_outputs = 10
    momentum = 0.9
    batch_size = 100
    n_epochs = 1

    X_train = tf.placeholder(dtype=tf.float32, shape=[None, image_size*image_size], name="X")
    y_train = tf.placeholder(dtype=tf.float32, shape=[None, n_outputs], name="y")

    training = tf.placeholder_with_default(False, shape=(), name="is_training")
    logits = model(X_train, y_train , 5, 100, training=training, dropout_rate=0.5, momentum=momentum)
    predictions = tf.nn.softmax(logits, name="preds")

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=logits, name="cross_entropy_loss"))

    with tf.name_scope("training"):
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
        saver = tf.train.Saver()

    with tf.name_scope("extra_operations"):  
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        clip_weights = tf.get_collection("max_norm")
        
    # extra config
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    ##for early stoping
    best_loss = np.infty
    checks_without_progress = 0
    max_checks_without_progress=20

    with tf.Session() as sess:
        
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)
        #initialize all variables
        sess.run(init)

        accuracy_per_epoch = []
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        t0 = time.time()
        
        for epoch in range(n_epochs):
            print("===========EPOCH: {} ===========".format(epoch))
            loss_per_epoch = []
            time_epoch= time.time()
            
            ##training
            for iteration in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                l,_, ex_op = sess.run([loss, training_op, extra_ops],
                                        feed_dict={X_train: batch[0].reshape(batch_size,image_size*image_size),
                                                    y_train: batch[1],
                                                    training: True,
                                                    learning_rate: 0.05})
                sess.run(clip_weights)
                loss_per_epoch.append(l)
            time_epoch_end = time.time()
            
            print('Average loss: {} in epoch: {}'.format(round(np.mean(loss_per_epoch),5), epoch))
            avg_loss = np.mean(loss_per_epoch)
            
            ##validation:
            valid_accuracy_per_epoch = []
            
            for i in range(mnist.train.num_examples//batch_size):
                valid_images,valid_labels = mnist.validation.next_batch(batch_size)
                valid_batch_predictions = sess.run(
                    predictions, feed_dict={X_train: valid_images.reshape(batch_size,image_size*image_size)})
                valid_accuracy_per_epoch.append(accurancy(valid_batch_predictions,valid_labels))
        
            mean_v_acc = np.mean(valid_accuracy_per_epoch)
            print('Average Valid accurancy: {} in epoch: {}'.format(round(np.mean(valid_accuracy_per_epoch),5), epoch))
            
            ## testing
            accuracy_per_epoch = []

            for i in range(mnist.test.labels.shape[0]//batch_size):
                test_images, test_labels = mnist.test.next_batch(batch_size)
                test_batch_predictions = sess.run(
                    predictions,feed_dict={X_train: test_images.reshape(batch_size,image_size*image_size)}
                )
                accuracy_per_epoch.append(accurancy(test_batch_predictions,test_labels))

            print('Average test accurancy: {} in epoch: {}'.format(round(np.mean(accuracy_per_epoch),5), epoch))
            avg_test_accuracy = np.mean(accuracy_per_epoch)
            
            delta_epoch_time = time_epoch_end-time_epoch

            print('Epoch time [s]: {}'.format(round(delta_epoch_time, 5)))

            # Execute the summaries
            summary = sess.run(merged_summaries, feed_dict={loss_summary_ph:avg_loss, accuracy_summary_ph:avg_test_accuracy})

            # Write the obtained summaries to the file, so it can be displayed in the TensorBoard
            summary_writer.add_summary(summary, epoch)
            
            #sAving 
            if epoch%10==0:
                saver.save(sess, checkpoint_path)
        
        
            #early stoping    
            loss_val = sess.run(loss, feed_dict={X_train: mnist.validation.images,
                                                y_train: mnist.validation.labels})
            
            if loss_val < best_loss:
                best_loss = loss_val
                checks_without_progress = 0
            else:
                checks_without_progress += 1
                if checks_without_progress > max_checks_without_progress:
                    print("Early stopping!!")
                    break
        
        time_end = time.time()  
        print(f"<=========== END: time of evaluation: {round(time_end-t0,4)} [s] ===========>")  
        
        sess.close()    
            
            


    
