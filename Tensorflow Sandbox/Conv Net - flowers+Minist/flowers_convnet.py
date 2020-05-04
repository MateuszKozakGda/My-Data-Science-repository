import os
import tensorflow as tf
import sys
import tarfile
from six.moves import urllib
from collections import defaultdict
from random import sample
import matplotlib.image as mpimg
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score
from datetime import datetime
from random import shuffle


AUTOTUNE = tf.data.experimental.AUTOTUNE
FLOWERS_URL_DATASET = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH2 = os.path.join("zestawy_danych", "kwiaty")

##hiding tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def fetch_flowers(url=FLOWERS_URL_DATASET, path=FLOWERS_PATH2):
    """
    Download and Fetch flowers dataset
    """
    if os.path.exists(FLOWERS_PATH2):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)
    
#fetch_flowers()

def flower_classes(path=FLOWERS_PATH2):
    """
    Return of unique classes from dataset
    """
    flowers_root_path = os.path.join(path, "flower_photos")
    flower_classes = sorted([dirname for dirname in os.listdir(flowers_root_path)
                  if os.path.isdir(os.path.join(flowers_root_path, dirname))])
    return flower_classes

def paths_for_all_photos(path=FLOWERS_PATH2):
    
    image_paths = defaultdict(list)
    root_path = os.path.join(path, "flower_photos")
    #root_path = path
    for flower_class in flower_classes():
        image_dir = os.path.join(root_path, flower_class)
        for filepath in os.listdir(image_dir):
            if filepath.endswith(".jpg"):
                image_paths[flower_class].append(os.path.join(image_dir, filepath))
    for paths in image_paths.values():
        paths.sort()
        
    return image_paths

def image_manipulation(image, target_width, target_height, max_zoom=0.2, training=False):
    """
    Image resizing function. 
    If training is set to True, function will rotate the image which is suitable for training. 
    For testing, training paramter should be kept as default.
    """
    ##croping image - trying to find rectangle which best fits target size of image
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)
    crop_height = tf.cond(crop_vertically,
                          lambda: width / target_image_ratio,
                          lambda: height)
    
    #Shrinking image rectangle by random coefficient -> coefficient belongs to random uniform distribution (1,1+max_zoom)
    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3])   # 3 = number of channels
    
    #training image manipulation
    if training:
        #cuting image by define box
        image = tf.random_crop(image, box_size)
        #flipping left 50% chance
        image = tf.image.random_flip_left_right(image)
        #flipping up and down
        image = tf.image.random_flip_up_down(image)
        #changing brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
        image_batch = tf.expand_dims(image, 0)
        image_batch  = tf.image.resize_bilinear(image_batch , 
                                size=[target_height, target_width])
        image = image_batch[0] / 255 ## normalizing colors to 0,1 range
        return image
    
    else: ##for testing image will be centralized
        image_batch = tf.expand_dims(image, 0) # expanding to 4d tensor
        image_batch  = tf.image.resize_bilinear(image_batch , 
                                size=[target_height, target_width])
        image = image_batch[0] / 255 ## normalizing colors to 0,1 range
        return image

#flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes())}
#print(flower_class_ids)
def one_hot_encoding(input_data):
    """
    One hot encoding class labels to 0,1 arrays
    """
    array_len = len(input_data)
    array = np.arange(array_len)
    one_hot_array = np.zeros((array.size, array.max()+1))
    one_hot_array[np.arange(array.size),array] = 1
    return one_hot_array

def get_flowers_labels_dict(data=flower_classes(), one_hot=True):
    """
    Making dictionary with class names and labels used for net training.
    """
    if one_hot:
        one_hot_encode = list(one_hot_encoding(data)) 
        flower_class_ids = dict(zip(data, one_hot_encode))
        return flower_class_ids
    else:   
        flower_class_ids = {flower_class: index for index, flower_class in enumerate(data)}
        return flower_class_ids

def get_flowers_path_and_classes(data=get_flowers_labels_dict(), paths=paths_for_all_photos(), only_paths=True):
    """
    Adding path to flower class label.
    """
    flower_paths_and_classes = []
    if not only_paths:
        for flower_class, path in paths.items():
            for p in path:
                flower_paths_and_classes.append([p, data[flower_class]])
        return flower_paths_and_classes
    else:
        paths_list = []
        class_list = []
        for flower_class, path in paths.items():
            for p in path:
                paths_list.append(p)
                class_list.append(data[flower_class])
        
        list_of_valus = list(zip(paths_list, class_list))
        shuffle(list_of_valus)
        paths_list, class_list = zip(*list_of_valus)
        test_ratio = 0.2
        train_size = int(len(paths_list) * (1 - test_ratio))
        train_x, train_y = paths_list[:train_size], class_list[:train_size]
        test_x, test_y = paths_list[train_size:], class_list[train_size:]
        return [train_x, train_y], [test_x, test_y]
 
def make_batch2(X,y, target_height=224, target_width=224, CHANNELS=3):
    # Convert to Tensor
    #target_height=224
    #target_width=224
    #CHANNELS=3
    
    #imagepaths = str(X)
    imagepaths = tf.convert_to_tensor(X, dtype=tf.string)
    # Build a TF Queue, shuffle data
    
    labels = y
    # Read images from disk
    image = tf.read_file(imagepaths)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    choice = np.random.uniform(low=0, high=1)
    if choice > 0.5:
    # Resize images to a common size
        image_shape = tf.cast(tf.shape(image), tf.float32)
        height = image_shape[0]
        width = image_shape[1]
        image_ratio = width / height
        target_image_ratio = target_width / target_height
        crop_vertically = image_ratio < target_image_ratio
        crop_width = tf.cond(crop_vertically,
                            lambda: width,
                            lambda: height * target_image_ratio)
        crop_height = tf.cond(crop_vertically,
                            lambda: width / target_image_ratio,
                            lambda: height)
        
        resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + 0.2)
        crop_width = tf.cast(crop_width / resize_factor, tf.int32)
        crop_height = tf.cast(crop_height / resize_factor, tf.int32)
        box_size = tf.stack([crop_height, crop_width, 3]) 
        
        image = tf.random_crop(image, box_size)
        #flipping left 50% chance
        image = tf.image.random_flip_left_right(image)
        #flipping up and down
        image = tf.image.random_flip_up_down(image)
        #changing brightness
        image = tf.image.random_brightness(image, max_delta=0.2)
      
    image = tf.image.resize_images(image, [target_height, target_width])
    # Normalize
    image = image * 1.0/127.5 - 1.0
        
    return image, labels

def test_batch(X,y, target_height=224,  target_width=224, CHANNELS=3):
    imagepaths = tf.convert_to_tensor(X, dtype=tf.string)
    labels = y
    labels = tf.convert_to_tensor(labels, dtype=np.float32)
    image = tf.read_file(imagepaths)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
 
    #resize
    image = tf.image.resize_images(image, [target_height, target_width])
    # Normalize
    image = image * 1.0/127.5 - 1.0
    return image, labels
    
def leaky_relu(alpha=0.01):
    """
    Leaky relu actviation function.
    alpha - slop of function when input is below 0
    """
    def parametrized_leaky_relu(z, name=None):
        return tf.maximum(alpha * z, z, name=name)
    return parametrized_leaky_relu

def inception_module(input, 
                     num_filters_1x1, 
                     num_filters_3x3_reduced,
                     num_filters_3x3,
                     num_filters_5x5_reduced,
                     num_filters_5x5,
                     num_filters_pool,
                     kernel_init,
                     bias_init,
                     activation,
                     name=None):
    """
    Inception module for inception net v1
    """
    with tf.variable_scope(name):
        #1x1 branch layer -> keeping size of entrance 
        conv_1x1_branch = tf.layers.conv2d(input, filters=num_filters_1x1, kernel_size=(1,1),activation=activation, 
                                        kernel_initializer = kernel_init, bias_initializer = bias_init, padding='SAME')
        
        #3x3 layer with previous 1x1 reduction 
        conv_3x3_reduction = tf.layers.conv2d(input, filters=num_filters_3x3_reduced, kernel_size=(1,1),activation=activation, 
                                        kernel_initializer = kernel_init, bias_initializer = bias_init, padding='SAME')
        conv_3x3 = tf.layers.conv2d(conv_3x3_reduction, filters=num_filters_3x3, kernel_size=(3,3),activation=activation, 
                                        kernel_initializer = kernel_init, bias_initializer = bias_init, padding='SAME')
        
        #5x5 layer with previous 1x1 reduction 
        conv_5x5_reduction = tf.layers.conv2d(input, filters=num_filters_5x5_reduced, kernel_size=(1,1),activation=activation, 
                                        kernel_initializer = kernel_init, bias_initializer = bias_init, padding='SAME')
        conv_5x5 = tf.layers.conv2d(conv_5x5_reduction, filters=num_filters_5x5, kernel_size=(5,5),activation=activation, 
                                        kernel_initializer = kernel_init, bias_initializer = bias_init, padding='SAME')
        
        ##max pooling
        pooling = tf.nn.max_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding="SAME")
        conv_pooling = tf.layers.conv2d(pooling, filters=num_filters_pool, kernel_size=(1,1),activation=activation, 
                                        kernel_initializer = kernel_init, bias_initializer = bias_init, padding='SAME')
        
        #concat layers
        concatenate = tf.concat([conv_1x1_branch, conv_3x3, conv_5x5, conv_pooling], axis=3)
        
    return concatenate
    
def stem_block(input, activation, kernel_init, bias_init):
    """
    First layers of net - only used at the begining
    """
    with tf.variable_scope("STEM_BLOCK"):
        first_layer = tf.layers.conv2d(input, filters=64, kernel_size=(7,7), strides=(2,2), activation= activation, padding='SAME',
                                    kernel_initializer = kernel_init, bias_initializer = bias_init, name="7x7_stem")
        first_pooling = tf.nn.max_pool(first_layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME",
                                        name="first_pooling_stem")
        sec_layer = tf.layers.conv2d(first_pooling, filters=64, kernel_size=(1,1), strides=(2,2),activation= activation, padding='SAME',
                                    kernel_initializer = kernel_init, bias_initializer = bias_init, name="1x1_stem")
        third_layer = tf.layers.conv2d(sec_layer, filters=192, kernel_size=(3,3), strides=(1,1),activation= activation, padding='SAME',
                                    kernel_initializer = kernel_init, bias_initializer = bias_init, name="3x3_stem")
        second_pooling = tf.nn.max_pool(third_layer, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME",
                                        name="second_pooling_stem")
    return second_pooling
    
def side_branch_regulizer(input, number_of_classes, activation, kernel_init, bias_init, branch_number, training=False):
    """
    Side branch with auxiliary output - used for regulazrization
    """
    with tf.variable_scope("Side_branch"+str(branch_number)):
        shape = input.get_shape().as_list()[1]
        avg_pooling = tf.layers.average_pooling2d(input, pool_size=(shape,shape), strides=(3,3), name="Pooling_brnach"+str(branch_number))
        first_layer = tf.layers.conv2d(avg_pooling, filters=128, kernel_size=(1,1), activation= activation, padding='SAME',
                                    kernel_initializer = kernel_init, bias_initializer = bias_init)
        pool_flat = tf.layers.flatten(first_layer, name="FLATTEN")
        dense1 = tf.layers.dense(pool_flat, units=1024, activation=activation)
        drop_out = tf.layers.dropout(dense1, rate=0.7, training=training, name="Dropout_side_branch"+str(branch_number))
        dense2 = tf.layers.dense(drop_out, units=number_of_classes, name="auxiliary_output"+str(branch_number))
    return dense2
    
def max_pooling(input, branch_number):
    """
    Max Pooling layer - used between inception modules
    """
    with tf.variable_scope("Max_pooling_layer"+str(branch_number)):
        pooling = tf.nn.max_pool(input, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")
    return pooling

def net_output(input, number_of_classes, name, training=False):
    """
    Net output layers - used at the end, layer have got a global average pooling
    """
    with tf.variable_scope(name):
        shape = input.get_shape().as_list()[1]
        global_pooling = tf.layers.average_pooling2d(input, pool_size=(shape,shape), strides=(1,1), padding="VALID", name="GLOBAL_AVG_POOLING")
        squeeze = tf.reduce_mean(global_pooling, axis=[1,2])
        drop_out = tf.layers.dropout(squeeze, rate=0.4, training=training, name="Dropout_Output_layer")                                         
        dense = tf.layers.dense(drop_out, units=number_of_classes, name="Main_output")
    return dense
    
def inception_v1(input_batch, num_classes, config_dict, training=False):
    """
    Making an incpetion v1 net. Config is a dictionary which contains informations about
    net configuration -> especialy for inception modules. 
    config_dict = {"stem_block" : [activation, kernel_init, bias_init],
                   "inception_n" : [num_filters_1x1, num_filters_3x3_reduced, num_filters_3x3, num_filters_5x5_reduced,
                     num_filters_5x5,num_filters_pool,kernel_init,bias_init,activation, name=None],
                    "max_pool1" : branch_number],
                    "side_branch1" : [activation, kenrel_init, bias_init, branch_number]
                     
    }
    """
    with tf.variable_scope("inception_model"):
        for module in config_dict.keys():
            if "stem_block" in module:
                stem = stem_block(input_batch, config_dict[module][0], config_dict[module][1], config_dict[module][2])
            if "inception_1" in module:
                inception = inception_module(stem, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "inception_2" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "max_pool1" in module:
                inception= max_pooling(inception, config_dict[module][0])
            
            if "inception_3" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "side_branch1" in module:
                y_1 = side_branch_regulizer(inception, num_classes, config_dict[module][0], config_dict[module][1], 
                                            config_dict[module][2], config_dict[module][3], training)
                
            if "inception_4" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "inception_5" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "inception_6" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "side_branch2" in module:
                y_2 = side_branch_regulizer(inception, num_classes, config_dict[module][0], config_dict[module][1], 
                                            config_dict[module][2], config_dict[module][3], training)
            
            if "inception_7" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "max_pool2" in module:
                inception= max_pooling(inception, config_dict[module][0])
            
            if "inception_8" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            if "inception_9" in module:
                inception = inception_module(inception, config_dict[module][0], config_dict[module][1], config_dict[module][2],
                                            config_dict[module][3], config_dict[module][4], config_dict[module][5],
                                            config_dict[module][6], config_dict[module][7], config_dict[module][8],
                                            config_dict[module][9])
            
        y_3 = net_output(inception, num_classes, "Net_main_output", training)
            
    return y_1, y_2, y_3

def get_directories():
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    file_name = os.path.basename(__file__)
    root_logdir = "tf_flower_net/{}".format(file_name)
    logdir = "{}/{}_run_{}/".format(root_logdir, file_name , now)
    checkpoint_path = "{}/{}_run_{}/Minist_model_CONVNET.ckpt".format(root_logdir, file_name , now)
        
    return logdir, checkpoint_path

def get_tensorboard(logdir):
    #root_logdir = "tf_flower_net/{}".format(logdir)
    #logdir = "{}/{}_przebieg-{}/".format(root_logdir, file_name , now)
    os.system('tensorboard --logdir=' + logdir)

## RUN the Code
if __name__ == '__main__':
    ##input data - net config
    image_size = [224, 224, 3]
    n_classes = len(flower_classes())
    kernel_init = tf.contrib.layers.variance_scaling_initializer()
    bias_init = tf.random_uniform_initializer(-0.0005,0.0005)
    n_epochs = 1
    batch_size = 2
    
    ##early stopping
    checks_without_progress = 0
    best_loss = np.infty
    best_params = None
    early_stoping_rounds = 10
    
    config = ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    
    net_config = {
        "stem_block" : [leaky_relu(), kernel_init, bias_init],
        "inception_1" : [64, 96,128,16,32,32, kernel_init, bias_init, leaky_relu(), "inception_1"],
        "inception_2" : [128, 128,192,32,96,64, kernel_init, bias_init, leaky_relu(), "inception_2"],
        "inception_3" : [192, 96, 208, 16, 48, 64, kernel_init, bias_init, leaky_relu(), "inception_3"],
        "inception_4" : [160, 112, 224, 24, 64, 64, kernel_init, bias_init, leaky_relu(), "inception_4"],
        "inception_5" : [128, 128, 256, 24, 64, 64, kernel_init, bias_init, leaky_relu(), "inception_5"],
        "inception_6" : [112, 144, 288, 32, 64, 64, kernel_init, bias_init, leaky_relu(), "inception_6"],
        "inception_7" : [256, 160, 320, 32, 128, 128, kernel_init, bias_init, leaky_relu(), "inception_7"],
        "inception_8" : [256, 160, 320, 32, 128, 128, kernel_init, bias_init, leaky_relu(), "inception_8"],
        "inception_9" : [384, 192, 384, 48, 128, 128, kernel_init, bias_init, leaky_relu(), "inception_9"],
        "max_pool1" : str(1),
        "max_pool2" : str(2),
        "side_branch1" : [leaky_relu(), kernel_init, bias_init, 1],
        "side_branch2" : [leaky_relu(), kernel_init, bias_init, 2]
    }
    
    flowers_train, flowers_test = get_flowers_path_and_classes()

    test_set_lengh = len(flowers_test[0])
    
    with tf.name_scope("inputs"):    
        training_tf = tf.placeholder_with_default(False, shape=(), name="is_training")
        #reuse_tf = tf.placeholder_with_default(False, shape=(), name="train/test switch")
        X = tf.placeholder(tf.string, shape=[None,], name="features")
        y = tf.placeholder(tf.float32, shape=[None, n_classes], name="labels")
        
        train_dataset = tf.data.Dataset.from_tensor_slices((X,y)).map(make_batch2, num_parallel_calls=AUTOTUNE).shuffle(500).batch(batch_size).repeat()
        test_dataset = tf.data.Dataset.from_tensor_slices((X,y)).map(test_batch, num_parallel_calls=AUTOTUNE).shuffle(500).batch(2*batch_size)
        
        iter = tf.data.Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)
        X_tr, y_tr = iter.get_next()
        train_init_op = iter.make_initializer(train_dataset)
        test_init_op = iter.make_initializer(test_dataset)
        #X,y = tf.cond(training_tf, 
                      #lambda: make_batch2(get_flowers_path_and_classes(), batch_size, training=True), 
                      #lambda: make_batch2(get_flowers_path_and_classes(), batch_size, training=False))
    
    with tf.name_scope("model"):
        y1,y2,y_main = inception_v1(X_tr, n_classes, net_config, training=training_tf)
        
    with tf.name_scope("loss"):
        xentropy1 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=y1, name="cross_entropy1")
        xentropy2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=y2, name="cross_entropy2")
        xentropy_main = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_tr, logits=y_main, name="cross_entropy_main")
        
        loss1 = tf.reduce_mean(xentropy1)
        loss2 = tf.reduce_mean(xentropy2)
        loss_main = tf.reduce_mean(xentropy_main)
        
        summ_of_losses = loss1 + loss2 + loss_main
    
    with tf.name_scope("evaluation"):
        y_proba1 = tf.nn.softmax(y1, name="Y_proba1")
        y_proba2 = tf.nn.softmax(y2, name="Y_proba2")
        y_proba_main = tf.nn.softmax(y_main, name="Y_proba_main")
        
        argmax_prediction1 = tf.argmax(y_proba1, 1)
        argmax_prediction2 = tf.argmax(y_proba2, 1)
        argmax_prediction_main = tf.argmax(y_proba_main, 1)
        argmax_y = tf.argmax(y_tr, 1)
        
        accuracy1 = tf.reduce_mean(tf.cast(tf.equal(argmax_prediction1, argmax_y), tf.float32))
        accuracy2 = tf.reduce_mean(tf.cast(tf.equal(argmax_prediction2, argmax_y), tf.float32))
        accuracy_main = tf.reduce_mean(tf.cast(tf.equal(argmax_prediction_main, argmax_y), tf.float32))
        
        acc_mean = (accuracy1+accuracy2+accuracy_main)/3
        
    with tf.name_scope("training"):
        init_rate = 0.001
        decay_steps = 10000
        decay_rate = 0.92
        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(init_rate, global_step, decay_steps, decay_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.95, use_nesterov=True)
        training_op = optimizer.minimize(summ_of_losses, global_step=global_step)
        
    with tf.name_scope("performance"):
        #loss summary
        loss_summary_ph3 = tf.placeholder(dtype=tf.float32, shape=None, name="loss_summary3")
        loss_summary3 = tf.summary.scalar("Loss_output", loss_summary_ph3)
        loss_summary_mean = tf.placeholder(dtype=tf.float32, shape=None, name="loss_summary_main")
        loss_summary_m = tf.summary.scalar("Loss_mean", loss_summary_mean)  
        #accurancy summary
        accuracy_summary_ph = tf.placeholder(tf.float32,shape=None, name='accuracy_summary_b1')
        accuracy_summary = tf.summary.scalar('accuracy_b1', accuracy_summary_ph)
        accuracy_summary_ph2 = tf.placeholder(tf.float32,shape=None, name='accuracy_summary_b2')
        accuracy_summary2 = tf.summary.scalar('accuracy_b2', accuracy_summary_ph2)
        accuracy_summary_ph3 = tf.placeholder(tf.float32,shape=None, name='accuracy_summary_b3')
        accuracy_summary3 = tf.summary.scalar('accuracy_b3', accuracy_summary_ph2) 
        accuracy_summary_ph4 = tf.placeholder(tf.float32,shape=None, name='accuracy_summary_mean')
        accuracy_summary4 = tf.summary.scalar('accuracy_mean', accuracy_summary_ph2)
        #recall summary 
        recall_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="recall_summary")
        recall_summary = tf.summary.scalar('recall', recall_summary_ph)
        #precision symmary
        precision_summary_ph = tf.placeholder(dtype=tf.float32, shape=None, name="recall_summary")
        precision_summary = tf.summary.scalar('precision', recall_summary_ph)
        #learning rate 
        learning_rate_ph = tf.placeholder(tf.float32,shape=None, name='learning_rate')
        lr_summary = tf.summary.scalar("learning_rate", learning_rate_ph)
        #merge
        merged_summaries = tf.summary.merge_all()
        
        
    with tf.name_scope("initialization_saver"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:
        #initialization
        init.run()
        #summary initializer
        log, checkpoint = get_directories()
        summary_writer = tf.summary.FileWriter(log, sess.graph)
        #list of eval variables ->validation
        var_list = [accuracy1, accuracy2, accuracy_main, acc_mean, loss_main, summ_of_losses, argmax_y, argmax_prediction_main]
        #set seed
        tf.random.set_seed(2)
        
        ###RUN iterator
        for epoch in range(n_epochs):
            #progress bar lengh
            sess.run(train_init_op, feed_dict = {X : flowers_train[0], y: flowers_train[1]})
            sess.run(training_tf, feed_dict={training_tf : True})          
            total_batch = int(len(flowers_train[1])/batch_size)
            bar = tqdm(range(total_batch), ncols=120, ascii=True)
            loss_per_epoch = []
            accuracy_per_epoch = []
            loss_l1, loss_l2, acc_m = [], [], []
            
            lr = sess.run(learning_rate)
                
            print("Learnig rate at epoch :{} lr = {:.6f}".format(epoch, lr))
            
            for bacth in bar:
                sess.run(training_op)
                train_loss, acc, acc2 = sess.run([summ_of_losses, acc_mean, accuracy_main])
                l1, l2= sess.run([loss1, loss2])
                loss_per_epoch.append(train_loss)
                accuracy_per_epoch.append(acc)
                acc_m.append(acc2)
                loss_l1.append(l1)
                loss_l2.append(l2)
                
                ml1, ml2= np.mean(loss_l1), np.mean(loss_l2)
                
                bar.set_description("Epoch: {}, Training cost: {:.4f}, Training Acc mean: {:.2f}%, Training acc: {:.2f}%".format(
                        epoch, np.mean(loss_per_epoch), np.mean(accuracy_per_epoch)*100, np.mean(acc_m)*100))
            
            #validation
            print("\n")
            print("TESTING....")
            sess.run(test_init_op, feed_dict = {X : flowers_test[0], y: flowers_test[1]})
            sess.run(training_tf, feed_dict={training_tf : False})   
            acc1, acc2, acc3, acc4, l_eval1, l_eval2, rec, prec = [], [], [], [], [], [], [], []
            for test_batch in range(int(test_set_lengh/(batch_size*2))):
                acc1_e, acc2_e, acc3_e, acc4_e, l_eval1_e, l_eval2_e, arg_y, arg_pred = sess.run(var_list)
                #append to lists
                acc1.append(acc1_e)
                acc2.append(acc2_e)
                acc3.append(acc3_e)
                acc4.append(acc4_e)
                l_eval1.append(l_eval1_e)
                l_eval2.append(l_eval2_e)
                
                rec_calc = recall_score(arg_y, arg_pred, average="micro")
                prec_calc = precision_score(arg_y, arg_pred, average="micro")
                
                rec.append(rec_calc)
                prec.append(prec_calc)
                
            print("Epoch: {}, Test loss : {:.4f}, Test Accuracy: {:.2f}%, Recall: {:.4f}%, Precision: {:.4f}%"
                  .format(epoch, np.mean(l_eval2), np.mean(acc4)*100, np.mean(rec)*100, np.mean(prec)*100))
                
            summary = sess.run(merged_summaries, feed_dict={
                loss_summary_ph3 : np.mean(l_eval1),
                loss_summary_mean : np.mean(l_eval2),
                accuracy_summary_ph : np.mean(acc1)*100,
                accuracy_summary_ph2 : np.mean(acc2)*100,
                accuracy_summary_ph3 : np.mean(acc3)*100,
                accuracy_summary_ph4 : np.mean(acc4)*100,
                recall_summary_ph : np.mean(rec),
                precision_summary_ph : np.mean(prec),
                learning_rate_ph : lr
            })
            summary_writer.add_summary(summary, epoch)

        print("===== END OF TRAINING =====")
        print("SAVING MODEL at {}".format(checkpoint))
        saver.save(checkpoint)   
        #Open Tensorboard
        get_tensorboard(log)
        