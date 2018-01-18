from tensorflow.examples.tutorials.mnist import input_data
import pdb
import matplotlib.pyplot as plt
import numpy as np
import random

#def imshow(img):
#    plt.figure()
#    img = (img+1)/2
#    img = np.clip(img, 0, 1)
#    plt.imshow(img)

"""
An object that handles data input
"""
class mnistData(object):
    raw_image_shape = (28, 28, 1)
    inputShape = raw_image_shape
    numClasses = 10
    numClutterPallet = 10000
    #translateSize is a 2 tuple with height, width of canvas to put mnist digit on
    #Clutteted will clutter input image with random patches of other digits
    #TODO set random seed
    def __init__(self, path, translateSize=None, clutterImg=False, numClutterRange=(5, 7), clutterSize=(8, 8)):
        self.mnist = input_data.read_data_sets(path, one_hot=False)
        self.num_train_examples = self.mnist.train.num_examples
        self.translateSize = translateSize
        self.clutterImg = clutterImg
        self.numClutterRange = numClutterRange
        self.clutterSize = clutterSize

        self.test_images = self.mnist.test.images
        self.test_labels = self.mnist.test.labels
        if(self.translateSize is not None):
            self.inputShape = (translateSize[0], translateSize[1], 1)
            #If translated, do test data here first
            images = self.mnist.test.images
            self.test_images = self.translate(self.test_images)

        if(self.clutterImg):
            #Extract a set of (training) images to pull clutter from
            self.clutterPallet, _ = self.mnist.train.next_batch(self.numClutterPallet)
            #Reshape into (N, y, x, f)
            self.clutterPallet = np.reshape(self.clutterPallet, (self.numClutterPallet,) + self.raw_image_shape)
            #If cluttered, do test data here first
            self.test_images = self.clutter(self.test_images)

    #Takes images of size (sample, features, and place digit on random position on canvas)
    def translate(self, images):
        (numExamples, numFeatures) = images.shape
        r_images = np.reshape(images, (numExamples,) + self.raw_image_shape)
        out_images = np.zeros((numExamples,) + self.translateSize + (1,))
        for i in range(numExamples):
            #Random x and y position
            yPos = random.randint(0, self.translateSize[0] - self.raw_image_shape[0])
            xPos = random.randint(0, self.translateSize[1] - self.raw_image_shape[1])
            out_images[i, yPos:yPos+self.raw_image_shape[0], xPos:xPos+self.raw_image_shape[1], :] = r_images[i]

        return np.reshape(out_images, [numExamples, -1])

    #Adds clutter to a given image
    def clutter(self, images):
        (numExamples, numFeatures) = images.shape
        r_images = np.reshape(images, (numExamples,) + self.inputShape)
        #TODO time this and optimize if need be
        for e_idx in range(numExamples):
            #Generate random number of clutters
            numClutters = random.randint(self.numClutterRange[0], self.numClutterRange[1])
            for j in range(numClutters):
                #Generate random index from pallet
                c_idx = random.randint(0, self.numClutterPallet-1)
                palletImg = self.clutterPallet[c_idx]
                #Generate random locations for patch extraction
                p_idx_y = random.randint(0, self.raw_image_shape[0] - self.clutterSize[0])
                p_idx_x = random.randint(0, self.raw_image_shape[1] - self.clutterSize[1])
                #Extract patch
                patch = palletImg[p_idx_y:p_idx_y+self.clutterSize[0],
                                  p_idx_x:p_idx_x+self.clutterSize[1],
                                  :]

                #Generate random target locations for clutter
                target_idx_y = random.randint(0, self.translateSize[0] - self.clutterSize[0])
                target_idx_x = random.randint(0, self.translateSize[1] - self.clutterSize[1])
                #Place patch onto image via alpha blending
                r_images[e_idx,
                         target_idx_y:target_idx_y+self.clutterSize[0],
                         target_idx_x:target_idx_x+self.clutterSize[1],
                         :] *= (1-patch)
                r_images[e_idx,
                         target_idx_y:target_idx_y+self.clutterSize[0],
                         target_idx_x:target_idx_x+self.clutterSize[1],
                         :] += patch

                #TODO patch overlaps image. Either clutter first then place image, or place nonzero pixels only
                #Preferably nonzero pixels only
        #reshape images and return
        out_images = np.reshape(r_images, [numExamples, -1])
        return out_images


    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)
        if(self.translateSize is not None):
            images = self.translate(images)
        if(self.clutterImg):
            images = self.clutter(images)
        return (images, labels)

    def getTestData(self):
        return (self.test_images, self.test_labels)

    def getValData(self):
        images = self.mnist.validation.images
        labels = self.mnist.validation.labels
        return(images, labels)

if __name__ == "__main__":
    path = "/home/slundquist/mountData/datasets/mnist"
    obj = mnistObj(path, translateSize=(60, 60), clutterImg=True)

    for i in range(10):
        (data, gt) = obj.getData(4)
        plt.figure()
        r_data = np.reshape(data[0], (60, 60))
        plt.imshow(r_data, cmap="gray")
        plt.show()
        plt.close('all')


