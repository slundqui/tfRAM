from tensorflow.examples.tutorials.mnist import input_data
import pdb
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

#def imshow(img):
#    plt.figure()
#    img = (img+1)/2
#    img = np.clip(img, 0, 1)
#    plt.imshow(img)


class patchExtractor(object):
    def __init__(self, win_size, glimpse_scales, zeroPad=False):
        self.images_ph = tf.placeholder(tf.float32, shape=[None, None, None, None], name="inImage")
        self.loc_ph = tf.placeholder(tf.float32, shape=[None, 2], name="inLoc")
        self.win_size = win_size
        self.glimpse_scales = glimpse_scales
        self.zeroPad = zeroPad

        self.output = self.get_glimpse()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_glimpse(self):
        imgs = self.images_ph
        loc = self.loc_ph
        image_size = tf.shape(imgs)

        scale_loc = tf.clip_by_value(loc, -1., 1.)

        glimpse_imgs = []
        #TODO explicitly test this
        for i in range(self.glimpse_scales):
            if(i > 0):
                #Scale image down
                imgs = tf.image.resize_images(imgs, [image_size[1]//(2*i), image_size[2]//(2*i)])

            if(self.zeroPad):
                #Pad image with 0
                pad_width = int(np.ceil(np.max(self.win_size)/2))
                pad_img = tf.pad(imgs, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
                f_image_size = tf.cast(tf.shape(imgs), tf.float32)
                #Adjust locs to new image shape
                pad_loc_0 = scale_loc[:, 0] * f_image_size[1] / (f_image_size[1]+2*pad_width)
                pad_loc_1 = scale_loc[:, 1] * f_image_size[2] / (f_image_size[2]+2*pad_width)
                pad_loc = tf.stack([pad_loc_0, pad_loc_1], axis=1)
                #Extract glimpses
                single_glimpse = tf.image.extract_glimpse(pad_img,
                                        [self.win_size[0], self.win_size[1]], pad_loc)
            else:
                #Extract glimpses at various sizes
                single_glimpse = tf.image.extract_glimpse(imgs,
                                        [self.win_size[0], self.win_size[1]], scale_loc)
            glimpse_imgs.append(single_glimpse)

        #Concatenate glimpse imgs in feature dim
        return tf.concat(glimpse_imgs, axis=3)

    def __call__(self, images, loc):
        feed_dict = {self.images_ph: images, self.loc_ph:loc}
        return self.sess.run(self.output, feed_dict=feed_dict)


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
    def __init__(self, path, normalize=False, translateSize=None, clutterImg=False, numClutter=(5, 7), clutterSize=(8, 8), flatten=True, getGt=True, patch=False, patchSize=[12, 12], numPatchScales=3, zeroPad=False):
        self.mnist = input_data.read_data_sets(path, one_hot=False)
        self.normalize = normalize
        self.num_train_examples = self.mnist.train.num_examples
        self.translateSize = translateSize
        self.clutterImg = clutterImg
        self.numClutter = numClutter
        self.clutterSize = clutterSize
        self.flatten=flatten
        self.getGt=getGt
        self.patch=patch

        self.test_images = self.mnist.test.images
        numTestImgs = self.test_images.shape[0]
        self.test_images = np.reshape(self.test_images, (numTestImgs,) + self.raw_image_shape)
        self.test_labels = self.mnist.test.labels
        if(self.translateSize is not None):
            self.inputShape = (translateSize[0], translateSize[1], 1)
            #If translated, do test data here first
            self.test_images = self.translate(self.test_images)

        if(self.clutterImg):
            #Extract a set of (training) images to pull clutter from
            self.clutterPallet, _ = self.mnist.train.next_batch(self.numClutterPallet)
            #Reshape into (N, y, x, f)
            self.clutterPallet = np.reshape(self.clutterPallet, (self.numClutterPallet,) + self.raw_image_shape)
            #If cluttered, do test data here first
            self.test_images = self.clutter(self.test_images)

        if(self.patch):
            self.inputShape = (patchSize[0], patchSize[1], numPatchScales)
            self.extractPatch = patchExtractor(patchSize, numPatchScales, zeroPad)

        if(self.flatten):
            self.test_images = np.reshape(self.test_images, [numTestImgs, -1])

    #Takes images of size (sample, features, and place digit on random position on canvas)
    def translate(self, images):
        numExamples = images.shape[0]
        out_images = np.zeros([numExamples, self.translateSize[0], self.translateSize[1], self.raw_image_shape[2]])
        for i in range(numExamples):
            #Random x and y position
            yPos = random.randint(0, self.translateSize[0] - self.raw_image_shape[0])
            xPos = random.randint(0, self.translateSize[1] - self.raw_image_shape[1])
            out_images[i, yPos:yPos+self.raw_image_shape[0], xPos:xPos+self.raw_image_shape[1], :] = images[i]

        return out_images

    #Adds clutter to a given image
    def clutter(self, images):
        numExamples = images.shape[0]
        #TODO time this and optimize if need be
        for e_idx in range(numExamples):
            #Generate random number of clutters
            if(type(self.numClutter) is tuple or type(self.numClutter) is list):
                numClutters = random.randint(self.numClutter[0], self.numClutter[1])
            else:
                numClutters = self.numClutter

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
                images[e_idx,
                         target_idx_y:target_idx_y+self.clutterSize[0],
                         target_idx_x:target_idx_x+self.clutterSize[1],
                         :] *= (1-patch)
                images[e_idx,
                         target_idx_y:target_idx_y+self.clutterSize[0],
                         target_idx_x:target_idx_x+self.clutterSize[1],
                         :] += patch

                #TODO patch overlaps image. Either clutter first then place image, or place nonzero pixels only
                #Preferably nonzero pixels only
        #reshape images and return
        return images

    #Gets numExample images and stores it into an outer dimension.
    #This is what TF object calls to get images for training
    def getData(self, numExample):
        images, labels = self.mnist.train.next_batch(numExample)
        #Reshape into y, x, f
        images = np.reshape(images, (numExample,) + self.raw_image_shape)
        if(self.translateSize is not None):
            images = self.translate(images)
        if(self.clutterImg):
            images = self.clutter(images)
        if(self.flatten):
            images = np.reshape(images, [numExample, -1])

        if(self.patch):
            #Generate random locs between -1 and 1
            rand_locs = np.random.uniform(-1, 1, size=[numExample, 2])
            images = self.extractPatch(images, rand_locs)

        if(self.normalize):
            images = (images - np.mean(images))/(np.std(images) + 1e-6)

        if(self.getGt):
            return (images, labels)
        else:
            return images

    def getTestData(self):
        if(self.getGt):
            return (self.test_images, self.test_labels)
        else:
            return self.test_images
#
#    def getValData(self):
#        images = self.mnist.validation.images
#        labels = self.mnist.validation.labels
#        return(images, labels)

if __name__ == "__main__":
    path = "/home/slundquist/mountData/datasets/mnist"
    obj = mnistData(path, translateSize=(60, 60), clutterImg=True, flatten=False, patch=True, patchSize=[50, 50], zeroPad=True)

    for i in range(10):
        (data, gt) = obj.getData(4)
        plt.figure()
        r_data = data[0, :, :, 0]
        plt.imshow(r_data, cmap="gray")
        plt.figure()
        r_data = data[0, :, :, 1]
        plt.imshow(r_data, cmap="gray")
        plt.figure()
        r_data = data[0, :, :, 2]
        plt.imshow(r_data, cmap="gray")
        plt.show()
        plt.close('all')


