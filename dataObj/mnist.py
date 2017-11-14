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
class mnistObj(object):
    raw_image_shape = (28, 28, 1)
    inputShape = raw_image_shape
    numClasses = 10
    #translateSize is a 2 tuple with height, width of canvas to put mnist digit on
    #Clutteted will clutter input image with random patches of other digits
    def __init__(self, path, translateSize=None, cluttered=False):
        self.mnist = input_data.read_data_sets(path, one_hot=False)
        self.num_train_examples = self.mnist.train.num_examples
        self.translateSize = translateSize
        self.cluttered = cluttered

        self.test_images = self.mnist.test.images
        self.test_labels = self.mnist.test.labels
        if(self.translateSize is not None):
            self.inputShape = (translateSize[0], translateSize[1], 1)
            #If translated, do test data at first here
            images = self.mnist.test.images
            self.test_images = self.translate(self.test_images)

        if(self.cluttered):
            assert(self.translateSize is not None)
            #TODO
            print("Not implemented yet")
            assert(False)

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

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)
        if(self.translateSize is not None):
            images = self.translate(images)
        return (images, labels)

    def getTestData(self):
        return (self.test_images, self.test_labels)

    def getValData(self):
        images = self.mnist.validation.images
        labels = self.mnist.validation.labels
        return(images, labels)

if __name__ == "__main__":
    path = "/home/slundquist/mountData/datasets/mnist"
    obj = mnistObj(path, translateSize=(60, 60))

    for i in range(10):
        (data, gt) = obj.getData(4)
        plt.figure()
        r_data = np.reshape(data[0], (60, 60))
        plt.imshow(r_data)
        plt.show()
        plt.close('all')


