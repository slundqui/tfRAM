#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataObj.mnist import mnistObj
from dataObj.multithread import mtWrapper
import numpy as np
import pdb
from tf.convBaseline import convBaseline

#Get object from which tensorflow will pull data from
#TODO turning off shuffle results in the same image everytime
#TODO images getting scaled improperly on top
path = "/home/slundquist/mountData/datasets/mnist"
dataObj = mnistObj(path)

device = '/gpu:0'

class Params(object):
    ##Bookkeeping params
    #Base output directory
    out_dir            = "/home/slundquist/mountData/ram/"
    #Inner run directory
    run_dir            = out_dir + "/mono_conv/"
    tf_dir             = run_dir + "/tfout"
    #Save parameters
    ckpt_dir           = run_dir + "/checkpoints/"
    save_file          = ckpt_dir + "/save-model"
    save_period        = 50000
    #output plots directory
    plot_dir           = run_dir + "plots/"
    plot_period        = 1718
    eval_period        = 1718 # 1 epoch
    #Progress step
    progress           = 100
    #Controls how often to write out to tensorboard
    write_step         = 300
    #Flag for loading weights from checkpoint
    load               = False
    load_file          = ""
    #Device to run on
    device             = device

    #data params
    num_train_examples = dataObj.num_train_examples

    #RAM params
    win_size           = 10      #The size of each glimpse in pixels in both x and y dimension
    batch_size         = 32      #Batch size of training
    eval_batch_size    = 50      #Batch size of testing
    original_size      = dataObj.inputShape #Size of the input image in (y, x, f)

    num_filters        = 8       #number of conv filters
    conv_stride        = 5       #stride of conv layer
    num_fc_units       = 256     #number of fc units
    num_classes        = 10      #Number of output classes
    max_grad_norm      = 5.      #Clipping norm for gradient clipping

    num_steps          = 300001  #Number of total steps
    lr_start           = 1e-3    #Starting learning rate for lr decay
    lr_min             = 1e-4    #Minimum learning rate for lr decay
    lr_decay           = .97     #Learning rate decay multiplier

params = Params()
#dataObj = mtWrapper(dataObj, params.batch_size)


#Allocate tensorflow object
#This will build the graph
tfObj = convBaseline(params)
print("Done init")
tfObj.trainModel(dataObj)
print("Done run")

tfObj.closeSess()
