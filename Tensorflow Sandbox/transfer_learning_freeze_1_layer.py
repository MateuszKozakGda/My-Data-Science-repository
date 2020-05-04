import argparse
import tensorflow as tf
from sklearn.metrics import recall_score, precision_score, confusion_matrix
import numpy as np
import os
from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

# model parsing 
"""ap = argparse.ArgumentParser()
ap.add_argument("-l", "--load_model", required=True,
  help="path to model meta graph")
args = ap.parse_args()"""
## batch making function

def get_random_batch(X, y, batch_size):
    """
    Getting random mini batch of specififed size
    """
    random_index = np.random.randint(0, len(X), batch_size)
    X_batch = X[random_index]
    y_batch = y[random_index]
    return X_batch, y_batch
  
def accuracy_eval(y_pred, labels):
  """
  y_pred size = (N_samples x n_classes)
  labales -> true values, size = (N_samples x n_classes)
  """
  return np.sum(np.argmax(y_pred,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]


## check model structure
  
def get_indices(higher_then=4):
  from tensorflow.examples.tutorials.mnist import input_data
  mnist_index = input_data.read_data_sets('MNIST_data')
  y_tr = mnist_index.train.labels
  y_ts = mnist_index.test.labels
  y_valid = mnist_index.validation.labels
  
  lista_index_train = []
  lista_index_test = []
  lista_index_valid = []

  for ind in range(y_tr.shape[0]):
      if y_tr[ind] > higher_then:
          lista_index_train.append(ind)
  for ind in range(y_ts.shape[0]):
      if y_ts[ind] > higher_then:
          lista_index_test.append(ind)
  for ind in range(y_valid.shape[0]):
      if y_valid[ind] > higher_then:
          lista_index_valid.append(ind)    
                          
  return lista_index_train, lista_index_test, lista_index_valid

tr_ind, tst_ind, val_ind = get_indices(higher_then=4)
    
mnist = input_data.read_data_sets('MNIST_data')
X_train = mnist.train.images
y_train = mnist.train.labels

##taking 0-4 digits
X_train1 = mnist.train.images[tr_ind]
y_train1 = mnist.train.labels[tr_ind]
X_valid1 = mnist.validation.images[val_ind]
y_valid1 = mnist.validation.labels[val_ind]
X_test1 = mnist.test.images[tst_ind]
y_test1 = mnist.test.labels[tst_ind]

### geting smaller dataset -> number from 5 to 9, n=100 samples for training, n=30 for validation of each number!

def sample_n_instances_per_class(X, y, n=100):
  Xs, ys = [], []
  for label in np.unique(y):
      idx = (y == label)
      Xc = X[idx][:n]
      yc = y[idx][:n]
      Xs.append(Xc)
      ys.append(yc)
  return np.concatenate(Xs), np.concatenate(ys)

X_train2, y_train2 = sample_n_instances_per_class(X_train1, y_train1, n=100)
X_valid2, y_valid2 = sample_n_instances_per_class(X_valid1, y_valid1, n=30)

## onehot encoding of y labels:
def one_hot_encoding(y, classes=10):
    shape = (y.shape[0], classes)
    one_hot = np.zeros(shape)
    rows = np.arange(y.shape[0])
    one_hot[rows, y] = 1
    return one_hot

y_train2 = one_hot_encoding(y_train2)
y_valid2 = one_hot_encoding(y_valid2)


#reste graph
tf.reset_default_graph()  
## restoring the graph
#restore_saver = tf.compat.v1.train.import_meta_graph(args.load_model)
restore_saver = tf.compat.v1.train.import_meta_graph("./minit_checkpoint/minist_model.ckpt.meta")

"""for op in tf.get_default_graph().get_operations():
    print(op.name)"""
  
n_outputs = 5

X = tf.compat.v1.get_default_graph().get_tensor_by_name("X:0")
y = tf.placeholder(tf.int32, shape=(None, n_outputs), name="y")
training = tf.compat.v1.get_default_graph().get_tensor_by_name("is_training:0")
learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")

#y_proba = tf.compat.v1.get_default_graph().get_tensor_by_name("Y_proba:0")
out = tf.compat.v1.get_default_graph().get_tensor_by_name("model/layer10/activation10:0")
logits = tf.layers.dense(out, n_outputs,
                          kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                          name="logits")
y_proba = tf.nn.softmax(logits, name="y_proba")

xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits, name="cross_entropy")
loss = tf.reduce_mean(xentropy, name="loss") 

#define trainable variables

output_layer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="logits")
optimizer = tf.train.AdamOptimizer(learning_rate, name="Adam2")
training_op = optimizer.minimize(loss, var_list=output_layer_vars)
#print(output_layer_vars)

##acc metrics

argmax_prediction = tf.argmax(y_proba, 1)
argmax_y = tf.argmax(y, 1)
acc =  tf.reduce_mean(tf.cast(tf.equal(argmax_prediction, argmax_y), tf.float32))

#saver and init
init = tf.global_variables_initializer()
five_frozen_saver = tf.train.Saver()

##session start
n_epochs = 100
batch_size = 20

max_checks_without_progress = 20
checks_without_progress = 0
best_loss = np.infty
early_stopping_rounds = 50

##saving path

saving_path = "./minit_checkpoint/minist_model_5_9_new_layer"

with tf.Session() as sess:
  
    init.run()
    restore_saver.restore(sess, "./minit_checkpoint/minist_model.ckpt")
    for var in output_layer_vars:
        var.initializer.run()
        
    for epoch in range(n_epochs):
      total_batch = int(X_train2.shape[0]/batch_size)
      bar = tqdm(range(total_batch), ncols=120, ascii=True)
      
      loss_per_epoch = []
      acc_per_epoch = []
      
      for batch in bar:
        X_batch, y_batch = get_random_batch(X_train2, y_train2[:,5:], batch_size=batch_size)
        feed_dict = {X: X_batch, y: y_batch, learning_rate: 0.01, training:True}
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch, learning_rate: 0.01, training:True})
        
        train_loss, predictions =  sess.run([loss, y_proba], feed_dict={X: X_batch, y: y_batch})
        train_accurancy = accuracy_eval(predictions, y_batch)
        
        loss_per_epoch.append(train_loss)
        acc_per_epoch.append(train_accurancy)
        
        bar.set_description("Epoch: {}, Training cost: {:.6f} Traning acc: {:.3f}".format(
            epoch, np.mean(loss_per_epoch), np.mean(acc_per_epoch)))
      
      if X_valid2 is not None and y_valid2 is not None:
        feed_dict2 = {X: X_valid2, y: y_valid2[:,5:]}
        loss_val, prediction_val, acc_val = sess.run([loss, y_proba, acc],
                                        feed_dict=feed_dict2)
        acc_mean = np.mean(acc_val)
        
        argmax_pred, argmax_true = sess.run([argmax_prediction, argmax_y], feed_dict=feed_dict2)
        
        prec = precision_score(argmax_true, argmax_pred, average="micro")
        rec= recall_score(argmax_true, argmax_pred, average="micro")
        
        print(f"Valid Acc : {round(acc_mean,5)*100}, Recall: {round(rec,5)}, Precision: {round(prec, 5)}")
        
        if loss_val < best_loss:
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            
        if checks_without_progress > early_stopping_rounds:
          print("Early Stopping!")
          five_frozen_saver.save(sess, saving_path)
          break
        
        
        
    print("CONFUSION MATRIX")
    feed_dict3 = {X: X_valid2, y: y_valid2[:,5:]}
    argmax_pred_conf, argmax_true_conf = sess.run([argmax_prediction, argmax_y], feed_dict=feed_dict3)
    
    print(confusion_matrix(argmax_true_conf, argmax_pred_conf))
  