import pdb
import numpy as np
import tensorflow as tf
from tf.glimpse import GlimpseNet, LocNet

from tf.base import base
from tf.utils import weight_variable, bias_variable, loglikelihood
#import matplotlib.pyplot as plt

#Spatial transform network
class convBaseline(base):
    def buildModel(self):
        inputShape = self.params.original_size
        #Running on GPU
        with tf.device(self.params.device):
            with tf.name_scope("Input"):
                self.images = tf.placeholder(tf.float32,
                        shape=[None,
                            inputShape[0]*inputShape[1]*inputShape[2]],
                        name = "images")

                reshape_images = tf.reshape(self.images, [-1, inputShape[0], inputShape[1], inputShape[2]])
                tf.summary.image('images', reshape_images)

                self.varDict['images'] = self.images

                self.labels = tf.placeholder(tf.int64,
                        shape=[None],
                        name="labels")
                self.varDict['labels'] = self.labels

            #Build core network
            with tf.variable_scope('conv_net'):
                w_conv1 = weight_variable((self.params.win_size, self.params.win_size, inputShape[2], self.params.num_filters))
                b_conv1 = bias_variable((self.params.num_filters))
                conv1 = tf.nn.relu(tf.nn.conv2d(
                            input=reshape_images,
                            filter = w_conv1,
                            strides = [1, self.params.stride, self.params.stride, 1],
                            padding= "SAME") + b_conv1)

                [numSamples, ny, nx, nf] = tf.shape(conv1)
                reshape_conv1 = tf.reshape(conv1, [self.numSamples, ny*nx*nf])

                w_fc = weight_variable((ny*nx*nf, self.params.num_fc_units))
                b_fc = bias_variable((self.params.num_fc_units))
                fc = tf.nn.relu(tf.matmul(reshape_conv1, w_vc) + b_fc)

            #Classification network
            with tf.variable_scope('classification_net'):
                output = fc
                #Pass through classification network for reward
                w_logit = weight_variable((self.params.num_fc_units, self.params.num_classes))
                b_logit = bias_variable((self.params.num_classes,))
                logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
                self.softmax = tf.nn.softmax(logits)

            with tf.variable_scope('loss'):
                # cross-entropy.
                xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
                xent = tf.reduce_mean(xent)
                pred_labels = tf.argmax(logits, 1)

                #hybrid loss
                loss = xent
                grads = tf.gradients(loss, var_list)
                grads, _ = tf.clip_by_global_norm(grads, self.params.max_grad_norm)

                #Add to scalar tensorboard
                self.scalarDict['xent'] = xent
                self.scalarDict['loss'] = loss

            with tf.variable_scope('accuracy'):
                #
                self.injectBool = tf.placeholder_with_default(False, shape=(), name="injectBool")
                self.injectAcc = tf.placeholder_with_default(0.0, shape=None, name="injectAcc")

                #Calculate accuracy
                pred_labels = tf.argmax(self.softmax, axis=1)
                calc_acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, self.labels), tf.float32))
                accuracy = tf.cond(self.injectBool, lambda: self.injectAcc, lambda: calc_acc)
                self.scalarDict['accuracy'] = accuracy

            with tf.variable_scope('optimizer'):
                # learning rate
                global_step = tf.get_variable(
                    'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
                training_steps_per_epoch = self.params.num_train_examples // self.params.batch_size
                # decay per training epoch
                learning_rate = tf.train.exponential_decay(
                    self.params.lr_start,
                    global_step,
                    training_steps_per_epoch,
                    self.params.lr_decay,
                    staircase=True)
                learning_rate = tf.maximum(learning_rate, self.params.lr_min)
                opt = tf.train.AdamOptimizer(learning_rate)
                self.train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

                self.scalarDict['learning_rate'] = learning_rate

    def trainStep(self, step, dataObj):
        (images, labels) = dataObj.getData(self.params.batch_size)
        #Build feeddict
        feed_dict = {self.images: images, self.labels: labels, self.eval_ph:False}
        #Write flag
        #TODO see if you can put this into base
        if(step%self.params.write_step == 0):
            self.writeTrainSummary(feed_dict)
        #Run optimizers
        self.sess.run(self.train_op, feed_dict = feed_dict)

    def evalModelBatch(self, dataObj):
        #Do validation
        (all_images, all_labels) = dataObj.getTestData()
        self.evalSet(all_images, all_labels)
        #TODO also run val data

    def evalSet(self, allImages, allLabels):
        (numExamples, _) = allImages.shape
        assert(numExamples == allLabels.shape[0])
        #TODO remove this
        assert(numExamples % self.params.eval_batch_size == 0)
        steps_per_epoch = numExamples // self.params.eval_batch_size
        correct_count = 0
        idx = 0
        for test_step in range(steps_per_epoch):
            images = allImages[idx:idx+self.params.eval_batch_size]
            labels = allLabels[idx:idx+self.params.eval_batch_size]
            #Write summary on first step only
            correct_count += self.evalModel(images, labels)
            idx += self.params.eval_batch_size
        accuracy = float(correct_count) / numExamples
        #Eval with last set of images and labels, with the final accuracy
        self.evalModelSummary(images, labels, accuracy)

    def evalModelSummary(self, images, labels, injectAcc):
        # Duplicate M times
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])

        feed_dict = {self.images: images, self.labels: labels,
                     self.injectBool: True, self.injectAcc:injectAcc,
                     #self.eval_ph: True}
                     self.eval_ph: False}
        self.writeTestSummary(feed_dict)

    def evalModel(self, images, labels):
        #labels_bak = labels
        # Duplicate M times
        # This repeats each experiment 10 times with random conditions
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])
        #feed_dict = {self.images: images, self.labels: labels, self.eval_ph: True}
        feed_dict = {self.images: images, self.labels: labels, self.eval_ph: False}

        softmax_val = self.sess.run(self.softmax, feed_dict=feed_dict)

        #Find average of duplications
        #This is averaging across noisy locations
        #softmax_val = np.reshape(softmax_val, [self.params.M, -1, self.params.num_classes])
        #softmax_val = np.mean(softmax_val, 0)
        #Find label value

        pred_labels = np.argmax(softmax_val, 1).flatten()
        correct_count = np.sum(pred_labels == labels)
        return correct_count






