#import matplotlib
#matplotlib.use('Agg')
from data.mnist import mnistData
from data.multithread import mtWrapper
import numpy as np
import pdb

batch_size = 128
device = "/gpu:0"
mt = False

#Get object from which tensorflow will pull data from
#TODO cross validation
path = "/home/slundquist/mountData/datasets/mnist"

if(mt):
    #Make new class based on mnist class
    mt_mnistData = mtWrapper(mnistData, batch_size)
    #Instantiate class
    dataObj = mt_mnistData(path, translateSize=(60, 60))
else:
    dataObj = mnistData(path, translateSize=(60, 60))

#Load default params
from params.dram import DramParams
params = DramParams()

params.batch_size = batch_size

#Overwrite various params
params.device = device
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples

params.win_size = 12
params.glimpse_scales = 2
params.sensor_size = params.win_size**2 * params.glimpse_scales
params.num_steps = 2000001

from tf.DRAM import DRAM
for nglimpse in [4, 6, 8]:
    params.run_dir = params.out_dir + "/mono_dram_translated_nglimpse_" + str(nglimpse) + "/"
    params.num_glimpses = nglimpse

    #Allocate tensorflow object
    #This will build the graph
    tfObj = DRAM(params)
    print("Done init")

    tfObj.trainModel(dataObj)
    tfObj.evalModelBatch(dataObj, writeOut=True)
    print("Done run")
    tfObj.closeSess()

