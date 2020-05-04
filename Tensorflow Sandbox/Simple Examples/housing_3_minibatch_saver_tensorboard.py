"""import tensorflow as tf
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

#making date named files for tensorboard dictionary
from datetime import datetime

now = datetime.utcnow().strftime("%Y_%m_%d_%H:%M:%S")
file_name = os.path.basename(__file__)
root_logdir = "tf_dzienniki/{}".format(file_name)
logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name ,now)

#
def reset_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)
    
#rest graphs
reset_graph()
#tf.compat.v1.reset_default_graph()
#
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

##epoch definition and learning rate
n_epochs = 100
learning_rate = 0.002
batch_size = 1000

##data prep
housing = fetch_california_housing()
m, n = housing.data.shape

scaller = StandardScaler()
scalled_data = scaller.fit_transform(housing.data)

#number of batches
batches = int(np.ceil(m / batch_size))

#print(m,n)
scalled_housing_data_plus_bias = np.c_[np.ones((m,1)), scalled_data]

##graph definition
#creating placeholders

X = tf.placeholder(dtype = tf.float32, shape=[None, n+1], name="X") ## n+1 bo dorucilismy kolumne z obciazeniem
y = tf.placeholder(dtype = tf.float32, shape=[None, 1], name="y")
#
theta = tf.Variable(tf.random_uniform([n+1,1], -1.0, 1.0, seed=42), name="theta") #inicjalizacja w przedziale -1,1
y_pred = tf.matmul(X, theta, name="prognozy")
## Data for tensorboard
with tf.name_scope("strata") as scope:
    error = y_pred-y
    mse = tf.reduce_mean(tf.square(error), name="mse")
##    
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

## Data for tensorboard
mse_summary = tf.summary.scalar("MSE", mse)
file_writter = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
##minibacth splitter function

def fetch_batch(epoch, batch_index, batch_size, X_data, y_data):
    np.random.seed(epoch * batches + batch_index)  # nieukazane w książce
    indices = np.random.randint(m, size=batch_size)  # nieukazane
    X_batch = X_data[indices] # nieukazane
    y_batch = y_data.reshape(-1, 1)[indices] # nieukazane
    X_batch = X_batch.astype("float32")
    y_batch = y_batch.astype("float32")
    return X_batch, y_batch

##saving session    
save_path = "/tmp/moj_model4.ckpt"
is_file = os.path.isfile(save_path)

##inicjalizacja zmiennych w wezlach
init = tf.global_variables_initializer()
saver = tf.train.Saver()

##changing names for graph modules

with tf.compat.v1.Session() as sess:

    ## restoring session
    if is_file:
        saver.restore(sess, save_path)
    else:
        sess.run(init) ## inicjalizacja w sesji
    
    for epochs in range(n_epochs):
        epoch_cost=0
        for batch_index in range(batches):
            X_batch, y_batch = fetch_batch(epochs, batch_index, batch_size, scalled_housing_data_plus_bias, housing.target)
            #print(X_batch)
            ##updating tensorboard results
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epochs*batches + batch_index
                file_writter.add_summary(summary_str, step)
                
            ##Runnning session
            t,c = sess.run([training_op, mse], feed_dict={X: X_batch, y: y_batch})
            epoch_cost += c/batches
                  
        if epochs % 10 == 0:
            print(f"Epoka: {epochs}, MSE: +/- {round(epoch_cost,5)}")
            ##saving session
            print("Saving session after {} epochs".format(epochs))
            saver.save(sess, save_path)    
                                     
    best_theta = theta.eval()
    
file_writter.flush()
file_writter.close()

print(f"Theta ostateczne : {best_theta}")

#print(X.shape)"""
