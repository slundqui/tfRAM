import pdb
import numpy as np
import tensorflow as tf
from glimpse import GlimpseNet, LocNet

from base import base
#import matplotlib.pyplot as plt

#Spatial transform network
class RAM(base):
    def extractGlimpse(self, loc):

    #Builds the model. inMatFilename should be the vgg file
    def buildModel(self):
        #Running on GPU
        with tf.device(self.device):
            with tf.name_scope("Input"):
                self.images = tf.placeholder(tf.float32,
                        shape=[None,
                            self.inputShape[0]*self.inputShape[1]*inputShape[2]],
                        name = "images")
                self.labels = tf.placeholder(tf.float32,
                        shape=[None],
                        name="labels")
            #Build aux nets
            # Build the aux nets.
            with tf.variable_scope('glimpse_net'):
              gl = GlimpseNet(config, images_ph)
            with tf.variable_scope('loc_net'):
              loc_net = LocNet(config)


    #TODO this function can probably be moved to base, with explicit build_feed_dict method in subclass
    def trainStep(self, step, trainDataObj):
        data = trainDataObj.getData(self.batchSize)
        #Build feeddict
        feed_dict = {self.inputImage: data[0], self.gtImage: data[1]}
        #Write flag
        #TODO see if you can put this into base
        if(step%self.writeStep == 0):
            self.writeTrainSummary(feed_dict)
        #Run all optimizers
        self.sess.run([self.optimizerDs, self.optimizerGen], feed_dict=feed_dict)

    def evalModel(self, image, gt=None):
        if(gt is None):
            feed_dict = {self.inputImage: image}
        else:
            feed_dict = {self.inputImage: image, self.gtImage: gt}
            self.writeTestSummary(feed_dict)






