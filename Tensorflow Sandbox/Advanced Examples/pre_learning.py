import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from datetime import datetime

##hiding errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#reset graph
tf.reset_default_graph()  

## prepearing batch data
minist =input_data.read_data_sets('MNIST_data')
   
X_train1 = minist.train.images
y_train1 = minist.train.labels

X_valid = minist.validation.images
y_valid = minist.validation.labels

X_test = minist.test.images
y_test = minist.test.labels


def max_norm_regularizer(threshold=1, axes=1, name="max_norm",
                         collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights) ## ---> trzeba wywolac funckcja get_collection()!!!
        return None 
    return max_norm

def generate_batch(images, labels, batch_size):
    size1 = batch_size // 2
    size2 = batch_size - size1
    if size1 != size2 and np.random.rand() > 0.5:
        size1, size2 = size2, size1
    X = []
    y = []
    while len(X) < size1:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if rnd_idx1 != rnd_idx2 and labels[rnd_idx1] == labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([1])
    while len(X) < batch_size:
        rnd_idx1, rnd_idx2 = np.random.randint(0, len(images), 2)
        if labels[rnd_idx1] != labels[rnd_idx2]:
            X.append(np.array([images[rnd_idx1], images[rnd_idx2]]))
            y.append([0])
    rnd_indices = np.random.permutation(batch_size)
    return np.array(X)[rnd_indices], np.array(y)[rnd_indices]

def build_net(input_shape, n_layers, name, n_neurons=100):
    with tf.variable_scope(name):
        for i in range(n_layers):
            with tf.variable_scope(name+"_layer"+str(i+1)):
                if i == 0:
                    hidden = tf.layers.dense(input_shape, n_neurons,
                                             activation=tf.nn.elu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                             kernel_regularizer = max_norm_regularizer(threshold=1.0),
                                             use_bias=True,
                                             bias_initializer=tf.random_uniform_initializer(-0.001,0.001),
                                             name="hidden"+str(i+1))
                else:
                    hidden = tf.layers.dense(hidden, n_neurons,
                                             activation=tf.nn.elu,
                                             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                             kernel_regularizer = max_norm_regularizer(threshold=1.0),
                                             use_bias=True,
                                             bias_initializer=tf.random_uniform_initializer(-0.001,0.001),
                                             name="hidden"+str(i+1))
        return hidden

if __name__ == "__main__":
 
    # MNIST input shape
    n_inputs = 28 * 28 
    #making 2 entrances for network
    X = tf.placeholder(tf.float32, shape=(None, 2, n_inputs), name="X")
    X1, X2 = tf.unstack(X, axis=1) 
    #y is shape of 2 -> it will work as logistic regression output
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    
    #making networks
    with tf.name_scope("model"):
        net1 = build_net(X1, n_layers=10, name="DNN_A")
        net2 = build_net(X2, n_layers=5, name="DNN_B")
        #concat outputs
        dnn_conncat = tf.concat([net1,net2], axis=1)
        #adding new hidden layers with 10 neurons
        hidden = tf.layers.dense(dnn_conncat, 10,activation=tf.nn.elu,
                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                name="concat_hidden")

        logits = tf.layers.dense(hidden, units=1, kernel_initializer=tf.contrib.layers.variance_scaling_initializer(), name="logits")
        y_proba = tf.nn.sigmoid(logits, name="y_proba")
        #prediction of indentity of nets outputs
        y_pred = tf.cast(tf.greater_equal(logits, 0), tf.float32, name="y_pred")
        y_pred_correct = tf.equal(y_pred, y)
        accuracy = tf.reduce_mean(tf.cast(y_pred_correct, tf.float32))
        
    #cost functions
    with tf.name_scope("model_cost"):
        y_as_float = tf.cast(y, tf.float32)
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_as_float, logits=logits, name="xentropy")
        loss = tf.reduce_mean(xentropy, name="loss")
    #learning
    with tf.name_scope("leaning"):
        learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")
        momentum = tf.placeholder(dtype=tf.float32, shape=None, name="momentum")

        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        training_op = optimizer.minimize(loss, name="training")
        
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
        
        #merged_summaries = tf.summary.merge([loss_summary, accuracy_summary, recall_summary, precision_summary])
        merged_summaries = tf.summary.merge_all()
        
    with tf.name_scope("Initializer"):
        init = tf.global_variables_initializer()
        saver_2xDNN = tf.train.Saver()
    
    with tf.name_scope("extra_operations"):  
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        clip_weights = tf.get_collection("max_norm")
    
    ##mode params
    n_epochs = 100
    batch_size = 500
    n_samples = minist.train.num_examples
    saving_path = "./minit_pre_learning/pre_learning"

    max_checks_without_progress = 20
    checks_without_progress = 0
    best_loss = np.infty
    early_stopping_rounds = 20

    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    file_name = os.path.basename(__file__)
    root_logdir = "pre_learning/{}".format(file_name)
    tensorboard_logdir = "{}/{}_summary_{}/".format(root_logdir, file_name, now)

    with tf.Session() as sess:
        
        init.run()
        summary_writer = tf.summary.FileWriter(tensorboard_logdir, sess.graph)
        
        if os.path.isfile(saving_path):
            saver_2xDNN.restore(saving_path+".ckpt")
        
        for epoch in range(n_epochs):
            total_batch = int(X_train1.shape[0]/batch_size)
            bar = tqdm(range(total_batch), ncols=120, ascii=True)        
            loss_per_epoch = []
            acc_per_epoch = []
            
            for batch in bar:
                X_batch, y_batch = generate_batch(X_train1, y_train1, batch_size)
                feed_dict = {X: X_batch, y: y_batch, learning_rate:0.015, momentum:0.95}
                loss_tr, _ , acc_train = sess.run([loss, training_op, accuracy], feed_dict=feed_dict)
                sess.run(clip_weights)
                loss_per_epoch.append(loss_tr)
                acc_per_epoch.append(acc_train)
                
                bar.set_description("Epoch: {}, Training cost: {:.6f} Traning acc: {:.3f}%".format(
                    epoch, np.mean(loss_per_epoch), np.mean(acc_per_epoch)*100)) 
                
                avg_train_loss = np.mean(loss_tr)
                avg_train_acc = np.mean(acc_per_epoch)
                
            if X_valid is not None and y_valid is not None:
                X_batch_valid, y_batch_valid = generate_batch(X_valid, y_valid, X_valid.shape[0])
                feed_dict2 = {X: X_batch_valid, y: y_batch_valid}
                loss_valid, acc_valid, y_valid_pred = sess.run([loss, accuracy, y_pred], feed_dict=feed_dict2)
                
                recall = recall_score(y_batch_valid, y_valid_pred)
                precision = precision_score(y_batch_valid, y_valid_pred)
                
                print(f"Valid Acc : {round(acc_valid,3)*100}%, Recall: {round(recall,5)}, Precision: {round(precision, 5)}")
            
                if loss_valid < best_loss:
                    best_loss = loss_valid
                    checks_without_progress = 0
                else:
                    checks_without_progress += 1
                
                if checks_without_progress > early_stopping_rounds:
                    print("Early Stopping!")
                    saver_2xDNN.save(sess, saving_path)
                    break
            
            feed_dict_summaries = {loss_summary_ph : avg_train_loss,
                                accuracy_summary_ph : avg_train_acc,
                                val_loss_summary_ph : loss_valid,
                                val_accuracy_summary_ph : acc_valid,
                                recall_summary_ph : recall,
                                precision_summary_ph : precision}
            
            summary = sess.run(merged_summaries, feed_dict=feed_dict_summaries)
            summary_writer.add_summary(summary, epoch)
            
        #save
            if epoch %5 ==0:
                saver_2xDNN.save(sess, saving_path+".ckpt")
        
        saver_2xDNN.save(sess, saving_path)
            
        #open tensorboard
        os.system('tensorboard --logdir=' + tensorboard_logdir)
          

            
            
        
    