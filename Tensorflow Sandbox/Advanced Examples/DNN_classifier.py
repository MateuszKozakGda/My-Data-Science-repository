from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorboard import program
from tensorboard import default
import numpy as np
import os
import sys
from tqdm import tqdm
from datetime import datetime

##importing my max norm regulizer
from max_norm_regulizer import max_norm_regularizer


class DNN_classifier(BaseEstimator, ClassifierMixin):
    """
    DNN_classifier is an object which is compatible with sklearn classes/obejcts.
    It can be used for hyperparameter optimisation. 
    """

    def __init__(self, n_hidden_layers=5, n_neurons=5, validation_split = None, n_epochs=50, optimizer=tf.train.AdamOptimizer,
                 learning_rate =0.01, batch_size=100, activation=tf.nn.elu, kernel_initializer = tf.contrib.layers.variance_scaling_initializer(),
                 kernel_regulizer = None, use_bias = True, bias_initializer = tf.random_uniform_initializer(-0.001,0.001), 
                 batch_normalization=False, batch_norm_momentum=0.95, dropout=False, dropout_rate=0.5, early_stoping_rounds=20, random_state=2):
        """
        initialization of model parameters
        validation_split -> split factor for training dataset which will be used as validation data
        """
        self.n_hidden_layers = n_hidden_layers
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.batch_size =batch_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.kernel_regulizer = kernel_regulizer
        self.use_bias = use_bias
        self.bias_initializer = bias_initializer
        self.batch_normalization = batch_normalization
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.early_stoping_rounds = early_stoping_rounds
        self.random_state = random_state
        self._session = None
        if optimizer is not tf.train.MomentumOptimizer:
            self.optimizer = optimizer
        else:
           self.optimizer = tf.train.MomentumOptimizer
        
    def __build_model(self, inputs, n_outputs, training):
        """
        Buliding model layers with n_hidden_layers dense layers and n_neurons neurons number
        """
        
        self.weight_init = self.kernel_initializer
        self.bias_init = self.bias_initializer
        
        with tf.name_scope("model"):
        
            for i in range(self.n_hidden_layers):
                with tf.name_scope("layer"+str(i+1)):
                    if i == 0:
                        if self.use_bias: 
                            self.hidden = tf.layers.dense(inputs, self.n_neurons, name="input",
                                                    kernel_initializer=self.weight_init, use_bias=True, bias_initializer=self.bias_init, 
                                                    kernel_regularizer=self.kernel_regulizer)
                        else:
                            self.hidden = tf.layers.dense(inputs, self.n_neurons, name="input",
                                                kernel_initializer=self.weight_init, use_bias=False, bias_initializer=None, 
                                                kernel_regularizer=self.kernel_regulizer)
                        if self.batch_normalization:
                            self.hidden = tf.layers.batch_normalization(self.hidden,training=training, 
                                                                        momentum=self.batch_norm_momentum, name="batch_norm"+str(i+1))
                        self.hidden = self.activation(self.hidden, name="activation"+str(i+1))
                        if self.dropout:
                            self.hidden = tf.layers.dropout(self.hidden, self.dropout_rate, training=training)
                        
                    else:
                        if self.use_bias: 
                            self.hidden = tf.layers.dense(self.hidden, self.n_neurons, name="hidden"+str(i+1),
                                                    kernel_initializer=self.weight_init, use_bias=True, bias_initializer=self.bias_init, 
                                                    kernel_regularizer=self.kernel_regulizer)
                        else:
                            self.hidden = tf.layers.dense(self.hidden, self.n_neurons, name="hidden"+str(i+1),
                                                kernel_initializer=self.weight_init, use_bias=False, bias_initializer=None, 
                                                kernel_regularizer=self.kernel_regulizer)
                        if self.batch_normalization:
                            self.hidden = tf.layers.batch_normalization(self.hidden,training=training, 
                                                                        momentum=self.batch_norm_momentum, name="batch_norm"+str(i+1))
                        self.hidden = self.activation(self.hidden, name="activation"+str(i+1))
                        if self.dropout:
                            self.hidden = tf.layers.dropout(self.hidden, self.dropout_rate, training=training)

            #output layer:
            with tf.name_scope("Output"):    
                if self.use_bias: 
                    self.hidden = tf.layers.dense(self.hidden, n_outputs.shape[1], name="output",
                                                    kernel_initializer=self.weight_init, use_bias=True, bias_initializer=self.bias_init, 
                                                    kernel_regularizer=self.kernel_regulizer)
                else:
                    self.hidden = tf.layers.dense(self.hidden, n_outputs.shape[1], name="output",
                                        kernel_initializer=self.weight_init, use_bias=False, bias_initializer=None, 
                                        kernel_regularizer=self.kernel_regulizer)
            
        return self.hidden
        
    def __build_graph(self, n_inputs, n_outputs):
        """
        Build tensorflow computional graph
        """
        if self.random_state:
            tf.set_random_seed(self.random_state)
            np.random.seed(self.random_state)
            
        X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        y = tf.placeholder(tf.int32, shape=(None, n_outputs), name="y")
        training = tf.placeholder_with_default(False, shape=(), name="is_training")
        
        logits = self.__build_model(X, y, training=training)
        y_proba = tf.nn.softmax(logits, name="Y_proba")
        
        with tf.name_scope("loss"):
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits, name="cross_entropy")
            loss = tf.reduce_mean(xentropy, name="loss")
        
        with tf.name_scope("training"):
            learning_rate = tf.placeholder(dtype=tf.float32, shape=None, name="learning_rate")
            if self.optimizer != tf.train.MomentumOptimizer:
                optimizer = self.optimizer(learning_rate=learning_rate)
            else:
                optimizer = self.optimizer(learning_rate=learning_rate, momentum=0.9)
            training_op = optimizer.minimize(loss)
        
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
            
        with tf.name_scope("accurancy_metrics"):
            
            argmax_prediction = tf.argmax(y_proba, 1)
            argmax_y = tf.argmax(y, 1)
            #needed values for recall and precison calc
            TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
            TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
            FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
            FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
            
            """TP = tf.count_nonzero(y_proba * y, dtype=tf.float32)
            TN = tf.count_nonzero((y_proba - 1) * (y - 1), dtype=tf.float32)
            FP = tf.count_nonzero(y_proba * (y - 1), dtype=tf.float32)
            FN = tf.count_nonzero((y_proba - 1) * y, dtype=tf.float32)"""
            
            ##calcs:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            acc =  tf.reduce_mean(tf.cast(tf.equal(argmax_prediction, argmax_y), tf.float32))
            
        with tf.name_scope("initialization"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            
        with tf.name_scope("extra_operations"):  
            extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if self.kernel_regulizer == max_norm_regularizer():
                self._clip_weights = tf.get_collection("max_norm")
            
        ## variable sharing for graph computation periods
        self._X, self._y, self._training = X, y, training
        self._learning_rate = learning_rate
        self._y_proba, self._loss = y_proba, loss
        #self._training_op, self._accuracy = training_op, accuracy
        self._training_op = training_op
        self._init, self._saver = init, saver
        self._extra_ops = extra_ops
        self._loss_summary_ph, self._loss_summary = loss_summary_ph, loss_summary
        self._accuracy_summary_ph, self._accuracy_summary = accuracy_summary_ph, accuracy_summary
        self._recall_summary_ph , self._recall_summary  = recall_summary_ph, recall_summary
        self._precision_summary_ph, self._precision_summary = precision_summary_ph, precision_summary
        self._val_loss_summary_ph, self._val_loss_summary = val_loss_summary_ph, val_loss_summary
        self._val_accuracy_summary_ph, self._val_accuracy_summary = val_accuracy_summary_ph, val_accuracy_summary
        self._merged_summaries = merged_summaries
        ##eval metrics
        self._rec, self._prec, self._acc_formula = precision, recall, acc
          
    def close_session(self):
        if self._session:
            self._session.close()
            
    def _accuracy_eval(self, y_pred, labels):
        """
        y_pred size = (N_samples x n_classes)
        labales -> true values, size = (N_samples x n_classes)
        """
        return np.sum(np.argmax(y_pred,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]
        
    def _get_model_params(self):
        """
        Getting values of all variables after early stopping - it is faster then saving to file.
        """
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}

    def _restore_model_params(self, model_params):
        """
        Assigning variables to speciffied values -> faster then restoring model with tf.Saver() methods
        """
        gvar_names = list(model_params.keys()) #getting params names
        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")
                      for gvar_name in gvar_names} #assinging operations to nodes
        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
        self._session.run(assign_ops, feed_dict=feed_dict)        
        
    def _get_validation_set(self, X, y, validation_split=0.2, random_state=2):
        """
        Stratiffied sampling used for splitting train set to validation set -> used for validation od model
        """
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, test_size=validation_split, random_state = random_state)
       
        return self.X_train, self.X_val, self.y_train, self.y_val
    
    def _get_random_batch(self, X, y, batch_size):
        """
        Getting random mini batch of specififed size
        """
        self.random_index = np.random.randint(0, len(X), batch_size)
        self.X_batch = X[self.random_index]
        self.y_batch = y[self.random_index]
        return self.X_batch, self.y_batch
    
    def _get_directories(self):
        """
        Tensorboard and tf.Save() directories for files saving
        """
        self.now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
        self.file_name = os.path.basename(__file__)
        self.root_logdir = "tf_tensorboard_DNNclasiffier/{}".format(self.file_name)
        self.logdir = "{}/{}_przebieg-{}/".format(self.root_logdir, self.file_name , self.now)
        self.checkpoint_path = "{}/{}_przebieg-{}/Minist_model.ckpt".format(self.root_logdir, self.file_name , self.now)
        
        return self.logdir, self.checkpoint_path
         
    def fit(self, X, y):
        """
        Fitting model for training data. If validation split is defined, model will use early stoping!!
        """
        self.close_session()

        # defining inputs and outputs of model
        n_inputs = X.shape[1]
        #self.classes_ = np.unique(y)
        n_outputs = y.shape[1]
    
        
        #building graph
        self._graph = tf.Graph()
        with self._graph.as_default():
            self.__build_graph(n_inputs, n_outputs)
            
        ##splitting dataset           
        if self.validation_split:    
            self.X_train, self.X_valid, self.y_train, self.y_valid = self._get_validation_set(X, y, 
                                                                                              validation_split = self.validation_split, 
                                                                                              random_state=self.random_state)
            
        else:
            self.X_train, self.y_train = X, y
        #early stoping params    
        self.checks_without_progress = 0
        self.best_loss = np.infty
        self.best_params = None
            
        # Training model
        self._session = tf.Session(graph=self._graph)
        with self._session.as_default() as sess:
            #directories 
            self.tensorboard_logdir, self.saver_directory = self._get_directories()
            #summary writter
            self.summary_writer = tf.summary.FileWriter(self.tensorboard_logdir, sess.graph)

            
            #start session
            self._init.run()
            
            for epoch in range(self.n_epochs):
                #progress bar lengh
                self.total_batch = int(self.X_train.shape[0]/self.batch_size)
                self.bar = tqdm(range(self.total_batch), ncols=120, ascii=True)
                
                ##tensorboard variables
                self.loss_per_epoch = []
                self.accuracy_per_epoch = []
                self.accuracy_per_epoch_val = []
                
                for batch in self.bar:
                    X_batch, y_batch = self._get_random_batch(self.X_train, self.y_train, batch_size = self.batch_size)
                    feed_dict = {self._X: X_batch, self._y: y_batch, self._learning_rate: self.learning_rate}
                    if self.batch_normalization or self.dropout is not None:
                        feed_dict[self._training] = True
                    
                    if self._extra_ops:
                        sess.run(self._extra_ops, feed_dict=feed_dict)
                    
                    if self.kernel_regulizer == max_norm_regularizer():
                       sess.run(self._clip_weights, feed_dict=feed_dict) 
                     
                    sess.run(self._training_op, feed_dict=feed_dict)
                    
                    """self.train_loss, self.train_accurancy = sess.run([self._loss, self._accuracy],
                                                                     feed_dict=feed_dict)"""
                    self.train_loss, self._predictions =  sess.run([self._loss, self._y_proba], feed_dict=feed_dict)
                    self.train_accurancy = self._accuracy_eval(self._predictions, y_batch)
                    self.loss_per_epoch.append(self.train_loss)
                    self.accuracy_per_epoch.append(self.train_accurancy)
                    
                    self.bar.set_description("Epoch: {}, Training cost: {:.6f} Traning acc: {:.3f}".format(
                        epoch, np.mean(self.loss_per_epoch), np.mean(self.accuracy_per_epoch)))
                  
                self.avg_train_loss = np.mean(self.loss_per_epoch)
                self.avg_acc_train = np.mean(self.accuracy_per_epoch)
                      
                if self.X_valid is not None and self.y_valid is not None:
                    """self.loss_val, self.acc_val = sess.run([self._loss, self._accuracy],
                                                    feed_dict={self._X: self.X_valid, self._y: self.y_valid})"""
                    
                    self.loss_val, self.prediction_val = sess.run([self._loss, self._y_proba],
                                                    feed_dict={self._X: self.X_valid, self._y: self.y_valid})
                    self.val_acc = self._accuracy_eval(self.prediction_val, self.y_valid)
                    self.acc_val = np.mean(self.val_acc)
                    
                    self.val_loss, self.rec, self.prec, self.acc_form = sess.run([self._loss, self._rec, self._prec, self._acc_formula],
                                                    feed_dict={self._X: self.X_valid, self._y: self.y_valid})
                    
                    print(f"Valid Acc : {self.acc_form}")
                    if self.loss_val < self.best_loss:
                        self.best_params = self._get_model_params()
                        self.best_loss = self.loss_val
                        self.checks_without_progress = 0
                    else:
                        self.checks_without_progress += 1
                    
                    #set description of training/validation
                    self.bar.set_description("Epoch: {}, Training cost: {:.6f}, Validation cost: {:.6f}, Traning acc: {:.3f}, Validation acc:{:.3f}".format(
                        epoch, self.avg_train_loss, self.loss_val, self.avg_acc_train*100, self.acc_val*100))

                    if self.checks_without_progress > self.early_stoping_rounds:
                        print("Early Stopping!")
                        break
                    
                ##summary writting
                self.summary = sess.run(self._merged_summaries, feed_dict={self._loss_summary_ph : self.avg_train_loss, 
                                                                           self._accuracy_summary_ph : self.avg_acc_train,
                                                                           self._val_loss_summary_ph: self.val_loss,
                                                                           self._val_accuracy_summary_ph: self.acc_form,
                                                                           self._recall_summary_ph : self.rec,
                                                                           self._precision_summary_ph : self.prec})  
                self.summary_writer.add_summary(self.summary, epoch)
             
             # if early stopping works, return best params of model
            if self.best_params:
                self._restore_model_params(self.best_params)
            return self
        
    def predict_proba(self, X):
        if not self._session:
            raise NotFittedError("Sample %s has not been fitted yet!" % self.__class__.__name__)
        with self._session.as_default() as sess:
            return self._y_proba.eval(feed_dict={self._X: X})

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return np.array([[self.classes_[class_index]]
                         for class_index in class_indices], np.int32)

    def save(self, path):
        self._saver.save(self._session, path)
        
    def run_tensorboard(self):
        self.now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
        self.file_name = os.path.basename(__file__)
        self.root_logdir = "tf_tensorboard_DNNclasiffier/{}".format(self.file_name)
        os.system('tensorboard --logdir=' + self.root_logdir)
 
 
def reset_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()
tf.reset_default_graph()
           
if __name__ == '__main__':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    X_train = mnist.train.images
    y_train = mnist.train.labels

    dnn_clf = DNN_classifier(n_neurons=100, n_hidden_layers=10, validation_split =0.2, kernel_regulizer=max_norm_regularizer(), batch_normalization=True,
                            learning_rate=0.015, use_bias=True, batch_size=500, dropout=True, dropout_rate=0.4)
    dnn_clf.fit(X_train, y_train)
    dnn_clf.run_tensorboard()
            
       