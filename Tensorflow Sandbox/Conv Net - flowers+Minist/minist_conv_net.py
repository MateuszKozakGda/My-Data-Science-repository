import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import recall_score, precision_score

##hiding tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import mininst data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

##input data configuration
height = int(np.sqrt(mnist.train.images[0].shape[0]))
windth = height
n_inputs = height*windth
n_channels = 1 ## grey palette of collours

out_puts_size = mnist.train.labels.shape[1]
n_epochs = 10
batch_size = 200

##leaky relu function:
def leaky_relu(alpha=0.01):
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu

def get_tensorboard():
    file_name = os.path.basename(__file__)
    root_logdir = "tf_tensorboard_CONV_NET/{}".format(file_name)
    #logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name , now)
    os.system('tensorboard --logdir=' + root_logdir)


def get_directories():
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    file_name = os.path.basename(__file__)
    root_logdir = "tf_tensorboard_CONV_NET/{}".format(file_name)
    logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name , now)
    checkpoint_path = "{}/{}_przebieg-{}/Minist_model_CONVNET.ckpt".format(root_logdir, file_name , now)
        
    return logdir, checkpoint_path

#net configuration
config = {"conv1_fmaps": [32, 64],
          "conv1_ksize" : [3, 3],
          "conv1_stride" : [1, 2],
          "activation" : [leaky_relu(alpha=0.01), leaky_relu(alpha=0.01)],
          "dense_layers_size" : 64, 
          }

##utillis functions deffinitions
def my_conv_layer(input, filters, kernel_size, strides ,activation, name, padding="SAME", 
                  bias=False, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(), 
                  bias_initializer = tf.random_uniform_initializer(-0.001,0.001)):
    """
    Convolution layer definition
    """
    if bias:
        conv = tf.layers.conv2d(input, 
                                filters=filters,
                                kernel_size = kernel_size,
                                kernel_initializer=kernel_initializer,
                                use_bias=bias,
                                bias_initializer = bias_initializer, 
                                strides=strides,
                                activation=activation, 
                                padding=padding, 
                                name=name)
    else:
        conv = tf.layers.conv2d(input, 
                                filters=filters,
                                kernel_size = kernel_size,
                                kernel_initializer=kernel_initializer,
                                strides=strides,
                                activation=activation, 
                                padding=padding, 
                                name=name)
    return conv
    
def build_net(input, output, config):
    n_iterations = len(config["conv1_fmaps"]) 
    with tf.variable_scope("conv_layers"):
        for i in range(n_iterations):
            if i == 0:
                conv = my_conv_layer(input, 
                                    filters = config["conv1_fmaps"][i],
                                    kernel_size = config["conv1_ksize"][i],
                                    strides=config["conv1_stride"][i],
                                    activation = config["activation"][i],
                                    name = "input_layer"
                )
            else: 
                conv = my_conv_layer(conv, 
                                        filters = config["conv1_fmaps"][i],
                                        kernel_size = config["conv1_ksize"][i],
                                        strides=config["conv1_stride"][i],
                                        activation = config["activation"][i],
                                        name = "conv_"+str(i))
    with tf.variable_scope("max_pool"):
        pooling = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="Max_POOLING")
        pool_flat = tf.layers.flatten(pooling, name="FLATTEN")
    
    with tf.variable_scope("dense_layers"):
        dense = tf.layers.dense(pool_flat, config["dense_layers_size"], activation=leaky_relu(alpha=0.01), name="flatten_dense")
    
    with tf.variable_scope("output"):
        logits = tf.layers.dense(dense, output, name="output")
        Y_proba = tf.nn.softmax(logits, name="Y_proba")
        
    return logits, Y_proba
                                        
def get_random_batch(X, y, batch_size):
    """
    Getting random mini batch of specififed size
    """
    random_index = np.random.randint(0, len(X), batch_size)
    X_batch = X[random_index]
    y_batch = y[random_index]
    return X_batch, y_batch

##model initialize
if __name__ == '__main__':
    with tf.name_scope("inputs"):
        X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
        X_reshaped = tf.reshape(X, shape=[-1, height, windth, n_channels])
        y = tf.placeholder(tf.int32, shape=[None, out_puts_size], name="y")

    with tf.name_scope("model"):
        logits, Y_proba = build_net(X_reshaped, out_puts_size, config)

    with tf.name_scope("learning"):
        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits, name="cross_entropy")
        loss = tf.reduce_mean(xentropy, name="loss")
        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss)

    with tf.name_scope("evaluation"):
        argmax_prediction = tf.argmax(Y_proba, 1)
        argmax_y = tf.argmax(y, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(argmax_prediction, argmax_y), tf.float32))
        
    with tf.name_scope("performance"):
        #loss summary
        loss_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="loss_summary")
        loss_summary = tf.summary.scalar("Loss", loss_summary_ph) 
        #accurancy summary
        accuracy_summary_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary')
        accuracy_summary = tf.summary.scalar('accuracy', accuracy_summary_ph)
        #val loss
        val_loss_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="val_loss_summary")
        val_loss_summary = tf.summary.scalar("val_Loss", val_loss_summary_ph) 
        #val accurancy summary
        val_accuracy_summary_ph = tf.placeholder(tf.float32,shape=None, name='val_accuracy_summary')
        val_accuracy_summary = tf.summary.scalar('val_accuracy', val_accuracy_summary_ph)
        #recall summary 
        recall_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="recall_summary")
        recall_summary = tf.summary.scalar('recall', recall_summary_ph)
        #precision symmary
        precision_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="recall_summary")
        precision_summary = tf.summary.scalar('precision', recall_summary_ph)
        
        merged_summaries = tf.summary.merge_all()

    with tf.name_scope("initialization_saver"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    ##early stopping
    checks_without_progress = 0
    best_loss = np.infty
    best_params = None
    early_stoping_rounds = 10


# run model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.4

with tf.Session(config=config) as sess:
    
    tensorboard_logdir, saver_directory = get_directories()
    #summary writter
    summary_writer = tf.summary.FileWriter(tensorboard_logdir, sess.graph)
    #start session
    init.run()
    
    X_train, y_train = mnist.train.images, mnist.train.labels
    X_val, y_val = mnist.validation.images, mnist.validation.labels
    X_val, y_val = get_random_batch(X_val, y_val, 3300)
    X_test, y_test = mnist.test.images, mnist.test.labels
    
    for epoch in range(n_epochs):
        #progress bar lengh
        total_batch = int(X_train.shape[0]/batch_size)
        bar = tqdm(range(total_batch), ncols=120, ascii=True)
        
        loss_per_epoch = []
        accuracy_per_epoch = []
        accuracy_per_epoch_val = []
        
        for batch in bar:
            X_batch, y_batch = get_random_batch(X_train, y_train, batch_size)
            feed_dict = {X: X_batch, y:y_batch}
            
            sess.run(training_op, feed_dict=feed_dict)
            
            ##adding data for tensorboard
            
            train_loss, train_accurancy = sess.run([loss, accuracy], feed_dict=feed_dict)
            loss_per_epoch.append(train_loss)
            accuracy_per_epoch.append(train_accurancy)
            
            bar.set_description("Epoch: {}, Training cost: {:.6f} Traning acc: {:.5}%".format(
                epoch, np.mean(loss_per_epoch), np.mean(accuracy_per_epoch)*100))
            
        avg_train_loss = np.mean(loss_per_epoch)
        avg_acc_train = np.mean(accuracy_per_epoch)
        
        
        #validation
        loss_val, prediction_val = sess.run([loss, accuracy],
                                        feed_dict={X: X_val, y: y_val})
        
        ##recal and precision evaluation
        y_pred = sess.run(Y_proba, feed_dict={X: X_val, y: y_val})
        
        #argmax_pred, armax_true = np.
        argmax_pred, argmax_true = sess.run([argmax_prediction, argmax_y], 
                                                            feed_dict={X: X_val, y: y_val})      
        prec = precision_score(argmax_true, argmax_pred, average="micro")
        rec= recall_score(argmax_true, argmax_pred, average="micro")
        acc = np.mean(prediction_val)
        
        print(f"Valid Acc : {round(acc*100, 4)}%, RECALL :{round(rec, 5)}, PRECISION: {round(prec, 5)}")
        
        if loss_val < best_loss:
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
        

        if checks_without_progress > early_stoping_rounds:
            print("Early Stopping!")
            break

        ##summary writting
        summary = sess.run(merged_summaries, feed_dict={loss_summary_ph : avg_train_loss, 
                                                                    accuracy_summary_ph : avg_acc_train,
                                                                    val_loss_summary_ph: loss_val,
                                                                    val_accuracy_summary_ph: acc,
                                                                    recall_summary_ph : rec,
                                                                    precision_summary_ph : prec})  
        
get_tensorboard()
        

    

