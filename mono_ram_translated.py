#import matplotlib
#matplotlib.use('Agg')
from dataObj.mnist import mnistObj
from dataObj.multithread import mtWrapper
import numpy as np
import pdb

#Get object from which tensorflow will pull data from
#TODO cross validation
path = "/home/slundquist/mountData/datasets/mnist"
dataObj = mnistObj(path, translateSize=(60, 60))

#Load default params
from params.ram import RamParams
params = RamParams()

#Overwrite various params
params.device = '/gpu:0'
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples

params.win_size = 12
params.glimpse_scales = 3
params.sensor_size = params.win_size**2 * params.glimpse_scales
params.num_steps = 2000001
params.lr_decay = .999
params.lr_start = 1e-3


#dataObj = mtWrapper(dataObj, params.batch_size)

from tf.RAM import RAM
for nglimpse in [4, 6, 8]:
    params.run_dir = params.out_dir + "/mono_ram_translated_nglimpse_" + str(nglimpse) + "/"
    params.num_glimpses = nglimpse

    #Allocate tensorflow object
    #This will build the graph
    tfObj = RAM(params)
    print("Done init")

    tfObj.trainModel(dataObj)
    tfObj.evalModelBatch(dataObj, writeOut=True)
    print("Done run")
    tfObj.closeSess()

#Conv control
from params.conv import ConvParams
params = ConvParams()
#Overwrite various params
params.device = '/gpu:0'
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples
params.run_dir = params.out_dir + "/mono_conv_translated/"
params.num_steps = 2000001
params.lr_decay = .999
params.lr_start = 5e-3

from tf.convBaseline import convBaseline
tfObj = convBaseline(params)
tfObj.trainModel(dataObj)
tfObj.evalModelBatch(dataObj, writeOut=True)
print("Done run")
tfObj.closeSess()

#FC control
from params.fc import FcParams
params = FcParams()
#Overwrite various params
params.device = '/gpu:0'
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples
params.num_steps = 2000001
params.lr_decay = .999
params.lr_start = 5e-3

from tf.fcBaseline import fcBaseline
for hidden_units in [64, 256]:
    params.num_fc1_units = hidden_units
    params.num_fc2_units = hidden_units
    params.run_dir = params.out_dir + "/mono_fc_translated_" + str(hidden_units) + "/"
    tfObj = fcBaseline(params)
    tfObj.trainModel(dataObj)
    tfObj.evalModelBatch(dataObj, writeOut=True)
    print("Done run")
    tfObj.closeSess()



