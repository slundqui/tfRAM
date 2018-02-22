import matplotlib
matplotlib.use('Agg')

from data.mnist import mnistData
from data.multithread import mtWrapper
import numpy as np
import pdb
batch_size = 32
device = "/gpu:0"
mt = False

#Get object from which tensorflow will pull data from
#TODO cross validation
path = "/home/slundquist/mountData/datasets/mnist"

if(mt):
    #Make new class based on mnist class
    mt_mnistData = mtWrapper(mnistData, batch_size)
    #Instantiate class
    dataObj = mt_mnistData(path, normalize=True, translateSize=(100, 100), clutterImg=True, numClutter=8, flatten=False, getGt=False, patch=True, numPatchScales=4, zeroPad=True)
else:
    dataObj = mnistData(path, normalize=True, translateSize=(100, 100), clutterImg=True, numClutter=8, flatten=False, getGt=False, patch=True, numPatchScales=4, zeroPad=True)

#TODO change these params to be objects
params = {
    #Base output directory
    'outDir':          "/home/slundquist/mountData/tfSparseCode/",
    #Inner run directory
    'runDir':          "/lca_adam_mnist_cluttered_big_noedge/",
    'tfDir':           "/tfout",
    #Save parameters
    'ckptDir':         "/checkpoints/",
    'saveFile':        "/save-model",
    'savePeriod':      1000, #In terms of displayPeriod
    #output plots directory
    'plotDir':         "plots/",
    'plotReconPeriod':  1000*1000,
    'plotWeightPeriod': 1000*1000,
    #Progress step
    'progress':        1000,
    #Controls how often to write out to tensorboard
    'writeStep':       1000, #500,
    #Flag for loading weights from checkpoint
    'load':            False,
    'loadFile':        "/home/slundquist/mountData/tfSparseCode/lca_adam_seismic_ps1024_nf256_dyn_scale/checkpoints/save-model-2200500",
    #Device to run on
    'device':          device,
    #####FISTA PARAMS######
    'numIterations':   1000001,
    'displayPeriod':   1000,
    #Batch size
    'batchSize':       batch_size,
    #Learning rate for optimizer
    'learningRateA':   5e-4,
    'learningRateW':   1e-3,
    #Lambda in energy function
    'thresh':          .00055,
    #Number of features in V1
    'numV':            256,
    'fc':              True,
    #Stride not used with fc model
    'VStrideY':        1,
    'VStrideX':        1,
    #Patch size not used with fc model
    'patchSizeY':      1,
    'patchSizeX':      1,
    'inputMult':       .1,
}

from TFSparseCode.tf.lca_adam import LCA_ADAM

#Allocate tensorflow object
tfObj = LCA_ADAM(params, dataObj)
print("Done init")

tfObj.runModel()
print("Done run")

tfObj.closeSess()

