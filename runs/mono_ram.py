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
    win_size           = 8
    bandwidth          = win_size**2
    batch_size         = 32
    eval_batch_size    = 50
    loc_std            = 0.22
    original_size      = 28
    num_channels       = 1
    depth              = 1
    sensor_size        = win_size**2 * depth
    minRadius          = 8
    hg_size            = 128
    hl_size            = 128
    g_size             = 256
    cell_output_size   = 256
    loc_dim            = 2
    cell_size          = 256
    cell_out_size      = cell_size
    num_glimpses       = 6
    num_classes        = 10
    max_grad_norm      = 5.

    num_steps          = 100000
    lr_start           = 1e-3
    lr_min             = 1e-4

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
