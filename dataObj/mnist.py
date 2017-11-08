import json

from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
import pdb

import collections
import threading
import time

def imshow(img):
    plt.figure()
    img = (img+1)/2
    img = np.clip(img, 0, 1)
    plt.imshow(img)

"""
An object that handles data input
"""
class mnistObj(object):
    raw_image_shape = (28, 28, 1)
    inputShape = raw_image_shape
    numClasses = 10
    def __init__(self, path):
        self.mnist = input_data.read_data_sets(path, one_hot=False)

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)
        return (images, labels)

        #images = np.reshape(images, (numExample,) + self.raw_image_shape)

#class multithreadDressDs(object):
#    def loadData(self, batchSize, getMeta = False):
#        self.loadBuf = self.dataObj.getData(batchSize, getMeta)
#
#    def __init__(self, dataObj, batchSize, getMeta=False):
#        self.dataObj = dataObj
#        self.batchSize = batchSize
#        self.getMeta = getMeta
#        #This is needed by tf code
#        self.inputShape = dataObj.inputShape
#        self.gtShape = dataObj.gtShape
#        self.numExamples = dataObj.numExamples
#
#        #Start first thread
#        self.loadThread = threading.Thread(target=self.loadData, args=(self.batchSize, self.getMeta))
#        self.loadThread.start()
#
#    #This function doesn't actually need numExample and getMeta, but this api matches that of
#    #image. So all we do here is assert numExample and getMeta are the same
#    def getData(self, numExample, getMeta = False):
#        assert(numExample == self.batchSize)
#        assert(getMeta == self.getMeta)
#        #Block loadThread here
#        self.loadThread.join()
#        #Store loaded data into local variable
#        #This should copy, not access reference
#        returnBuf = self.loadBuf[:]
#        #Launch new thread to load new buffer
#        self.loadThread = threading.Thread(target=self.loadData, args=(self.batchSize, self.getMeta))
#        self.loadThread.start()
#        #Return stored buffer
#        return returnBuf


if __name__ == "__main__":
    path = "/home/slundquist/mountData/datasets/mnist"
    obj = mnistObj(path)

    for i in range(10):
        (data, gt) = obj.getData(4, False)
        pdb.set_trace()
        plt.figure()
        plt.imshow((data[0]+1)/2)
        plt.show()
        plt.close('all')


