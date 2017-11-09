from tensorflow.examples.tutorials.mnist import input_data

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
    def __init__(self, path):
        self.mnist = input_data.read_data_sets(path, one_hot=False)
        self.num_train_examples = self.mnist.train.num_examples


    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)
        return (images, labels)

    def getTestData(self):
        images = self.mnist.test.images
        labels = self.mnist.test.labels
        return(images, labels)

    def getValData(self):
        images = self.mnist.validation.images
        labels = self.mnist.validation.labels
        return(images, labels)

#if __name__ == "__main__":
#    path = "/home/slundquist/mountData/datasets/mnist"
#    obj = mnistObj(path)
#
#    for i in range(10):
#        (data, gt) = obj.getData(4, False)
#        pdb.set_trace()
#        plt.figure()
#        plt.imshow((data[0]+1)/2)
#        plt.show()
#        plt.close('all')


