#import matplotlib
#matplotlib.use('Agg')
from data.mnist import mnistData
from data.multithread import mtWrapper
import numpy as np
import pdb

batch_size = 32
device = "/gpu:1"
mt = False

#Get object from which tensorflow will pull data from
#TODO cross validation
path = "/home/slundquist/mountData/datasets/mnist"

if(mt):
    #Make new class based on mnist class
    mt_mnistData = mtWrapper(mnistData, batch_size)
    #Instantiate class
    dataObj = mt_mnistData(path)
else:
    dataObj = mnistData(path)

#Load default params
from params.ram import RamParams
params = RamParams()

#Overwrite various params
params.device = device
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples

#dataObj = mtWrapper(dataObj, params.batch_size)

from tf.RAM import RAM

#Loop through num_glimpses
for nglimpse in range(2, 8):
    params.run_dir = params.out_dir + "/mono_ram_base_nglimpse_" + str(nglimpse) + "/"
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
params.device = device
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples
params.run_dir = params.out_dir + "/mono_conv_base/"

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
params.device = device
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples
params.run_dir = params.out_dir + "/mono_fc_base/"

from tf.fcBaseline import fcBaseline
tfObj = fcBaseline(params)
tfObj.trainModel(dataObj)
tfObj.evalModelBatch(dataObj, writeOut=True)
print("Done run")
tfObj.closeSess()



