from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf.utils import weight_variable, bias_variable, batchnorm

import pdb

class GlimpseNet(object):
    """Glimpse network.

    Take glimpse location input and output features for RNN.

    """

    def __init__(self, config, images_ph):
        #Only allow square images for now
        self.original_size = config.original_size
        assert(self.original_size[0] == self.original_size[1])
        self.glimpse_scales = config.glimpse_scales
        self.sensor_size = config.sensor_size
        self.win_size = config.win_size
        #self.minRadius = config.minRadius
        #self.depth = config.depth

        self.hg_size = config.hg_size
        self.hl_size = config.hl_size
        self.g_size = config.g_size
        self.loc_dim = config.loc_dim
        self.loc_ratio = config.loc_pixel_ratio

        self.images_ph = images_ph

        self.init_weights()

    def init_weights(self):
        """ Initialize all the trainable weights."""
        self.w_g0 = weight_variable((self.sensor_size, self.hg_size))
        self.b_g0 = bias_variable((self.hg_size,))
        self.w_l0 = weight_variable((self.loc_dim, self.hl_size))
        self.b_l0 = bias_variable((self.hl_size,))
        self.w_g1 = weight_variable((self.hg_size, self.g_size))
        self.b_g1 = bias_variable((self.g_size,))
        self.w_l1 = weight_variable((self.hl_size, self.g_size))
        self.b_l1 = bias_variable((self.g_size,))

    def getVars(self):
        outDict = {'glimpsenet_w_g0': self.w_g0,
                   'glimpsenet_b_g0': self.b_g0,
                   'glimpsenet_w_l0': self.w_l0,
                   'glimpsenet_b_l0': self.b_l0,
                   'glimpsenet_w_g1': self.w_g1,
                   'glimpsenet_b_g1': self.b_g1,
                   'glimpsenet_w_l1': self.w_l1,
                   'glimpsenet_b_l1': self.b_l1}
        return outDict


    def get_glimpse(self, loc):
        """Take glimpse on the original images."""
        imgs = tf.reshape(self.images_ph, [
            tf.shape(self.images_ph)[0], self.original_size[0], self.original_size[1],
            self.original_size[2]
        ])

        scale_loc = tf.clip_by_value(loc * 2 * self.loc_ratio, -1., 1.)

        glimpse_imgs = []
        #TODO explicitly test this
        for i in range(self.glimpse_scales):
            if(i > 0):
                #Scale image down
                imgs = tf.image.resize_images(imgs, [self.original_size[0]//(2*i), self.original_size[1]//(2*i)])

            #Extract glimpses at various sizes
            single_glimpse = tf.image.extract_glimpse(imgs,
                                                    [self.win_size, self.win_size], scale_loc)
            glimpse_imgs.append(tf.reshape(single_glimpse, [
                tf.shape(scale_loc)[0], self.win_size * self.win_size * self.original_size[2]
            ]))

        #Concatenate glimpse imgs
        return tf.concat(glimpse_imgs, axis=1)

    def __call__(self, loc, is_train):
        glimpse_input = self.get_glimpse(loc)
        glimpse_input = tf.reshape(glimpse_input,
                                   (tf.shape(loc)[0], self.sensor_size))

        #G pipeline, which encodes glimpse
        g = tf.nn.relu(tf.nn.xw_plus_b(glimpse_input, self.w_g0, self.b_g0))
        #g = batchnorm(g, "gn_g0", is_train)
        g = tf.nn.xw_plus_b(g, self.w_g1, self.b_g1)

        #L pipeline, which encode locations
        l = tf.nn.relu(tf.nn.xw_plus_b(loc, self.w_l0, self.b_l0))
        #l = batchnorm(l, "gn_l0",is_train)
        l = tf.nn.xw_plus_b(l, self.w_l1, self.b_l1)

        #Combine
        g = tf.nn.relu(g + l)
        #Batch norm
        #g = batchnorm(g, "gn_g1", is_train)
        return g


class LocNet(object):
    """Location network.

    Take hidden state from core LSTM and calculates next location

    """

    def __init__(self, config):
        self.loc_dim = config.loc_dim
        self.input_dim = config.cell_size
        self.loc_std = config.loc_std
        self.loc_ratio = config.loc_pixel_ratio

        self.init_weights()

    def init_weights(self):
        self.w = weight_variable((self.input_dim, self.loc_dim))
        self.b = bias_variable((self.loc_dim,))

    def getVars(self):
        outDict = {'locnet_w': self.w,
                   'locnet_b': self.b,
                   }
        return outDict


    def __call__(self, input):
        mean = tf.nn.xw_plus_b(input, self.w, self.b)
        mean = tf.clip_by_value(mean, -1.0/(2*self.loc_ratio), 1.0/(2*self.loc_ratio))

        #Adds random noise to the location for training
        train_loc = mean + tf.random_normal(
            (tf.shape(input)[0], self.loc_dim), stddev=self.loc_std)

        loc = tf.stop_gradient(train_loc)

        #Set output location with no noise
        #eval_loc = mean

        #Select train or eval based on eval_ph
        #loc = tf.where(eval_ph, eval_loc, train_loc)

        #No backprop from location network
        #loc = tf.stop_gradient(loc)
        return loc, mean

