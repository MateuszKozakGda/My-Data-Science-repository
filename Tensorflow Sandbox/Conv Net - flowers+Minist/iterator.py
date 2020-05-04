import os
from datetime import datetime

def get_directories():
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    file_name = os.path.basename("huj")
    root_logdir = "tf_flower_net/{}".format(file_name)
    logdir = "{}/{}_run_{}/".format(root_logdir, file_name , now)
    checkpoint_path = "{}/{}_run_{}/Minist_model_CONVNET.ckpt".format(root_logdir, file_name , now)
    
    return logdir, checkpoint_path
