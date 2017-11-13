#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataObj.mnist import mnistObj
from dataObj.multithread import mtWrapper
import numpy as np
import pdb
from tf.RAM import RAM

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
    run_dir            = out_dir + "/mono_ram/"
    tf_dir             = run_dir + "/tfout"
    #Save parameters
    ckpt_dir           = run_dir + "/checkpoints/"
    save_file          = ckpt_dir + "/save-model"
    save_period        = 10000
    #output plots directory
    plot_dir           = run_dir + "plots/"
    plotPeriod         = 1718
    eval_period        = 1718 # 1 epoch
    #Progress step
    progress           = 100
    #Controls how often to write out to tensorboard
    write_step         = 100
    #Flag for loading weights from checkpoint
    load               = False
    load_file          = ""
    #Device to run on
    device             = device

    #data params
    num_train_examples = dataObj.num_train_examples

    #RAM params
    win_size           = 8       #The size of each glimpse in pixels in both x and y dimension
    batch_size         = 32      #Batch size of training
    eval_batch_size    = 50      #Batch size of testing
    loc_std            = 0.22    #Standard deviation of random noise added to locations
    original_size      = dataObj.inputShape #Size of the input image in (y, x, f)
    num_channels       = 1       #Number of channels in input image
    sensor_size        = win_size**2 * num_channels #Total size of input glimpse
    hg_size            = 128     #Number of features in first layer for glimpse encode
    hl_size            = 128     #Number of features in first layer for location encode
    g_size             = 256     #Number of features in second layer (combine g and l)
    loc_dim            = 2       #Number of dimensions used in the location output
    cell_size          = 256     #Size of hidden latent space in LSTM
    num_glimpses       = 6       #Number of total glimpses
    num_classes        = 10      #Number of output classes
    max_grad_norm      = 5.      #Clipping norm for gradient clipping

    num_steps          = 100000  #Number of total steps
    lr_start           = 1e-3    #Starting learning rate for lr decay
    lr_min             = 1e-4    #Minimum learning rate for lr decay

    # Monte Carlo sampling
    M                  = 10

params = Params()
#dataObj = mtWrapper(dataObj, params.batch_size)


#Allocate tensorflow object
#This will build the graph
tfObj = RAM(params)

print("Done init")
tfObj.trainModel(dataObj)
print("Done run")

tfObj.closeSess()
