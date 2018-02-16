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
    dataObj = mt_mnistData(path, translateSize=(60, 60), clutterImg=True, numClutter=4)
else:
    dataObj = mnistData(path, translateSize=(60, 60), clutterImg=True, numClutter=4)

#Conv control
from params.conv import ConvParams
params = ConvParams()
#Overwrite various params
params.device = device
params.original_size = dataObj.inputShape
params.num_train_examples = dataObj.num_train_examples
params.run_dir = params.out_dir + "/conv_cluttered/"
params.num_steps = 2000001
params.lr_decay = .999
params.lr_start = 1e-3

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
params.num_steps = 2000001
params.lr_decay = .999
params.lr_start = 1e-3

from tf.fcBaseline import fcBaseline
for hidden_units in [64, 256]:
    params.num_fc1_units = hidden_units
    params.num_fc2_units = hidden_units
    params.run_dir = params.out_dir + "/fc_cluttered_" + str(hidden_units) + "/"
    tfObj = fcBaseline(params)
    tfObj.trainModel(dataObj)
    tfObj.evalModelBatch(dataObj, writeOut=True)
    print("Done run")
    tfObj.closeSess()



