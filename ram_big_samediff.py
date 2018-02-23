#import matplotlib
#matplotlib.use('Agg')
from data.mnist import SDMnistData
from data.multithread import mtWrapper
import numpy as np
import pdb

batch_size = 32
device = "/gpu:0"

#Get object from which tensorflow will pull data from
#TODO cross validation
path = "/home/slundquist/mountData/datasets/mnist"

dataObj = SDMnistData(path, translateSize=(100, 100))

#Load default params
from params.ram import RamParams
params = RamParams()

params.batch_size = batch_size

#Overwrite various params
params.device = device
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples

params.win_size = 12
params.glimpse_scales = 4
params.sensor_size = params.win_size**2 * params.glimpse_scales
params.num_classes = dataObj.numClasses
params.reinforce_lambda = 1
params.lr_start = 1e-4
params.loc_std = 0.1
params.num_steps = 10000001  #Number of total steps


from tf.RAM import RAM
for nglimpse in [8]:
    params.run_dir = params.out_dir + "/ram_big_nglimpse_" + str(nglimpse) + "_samediff/"
    params.num_glimpses = nglimpse

    #Allocate tensorflow object
    #This will build the graph
    tfObj = RAM(params)
    print("Done init")

    tfObj.trainModel(dataObj)
    tfObj.evalModelBatch(dataObj, writeOut=True)
    print("Done run")
    tfObj.closeSess()

