import pdb
import numpy as np
import tensorflow as tf

from tf.base import base
from tf.utils import weight_variable, bias_variable
#import matplotlib.pyplot as plt

#Spatial transform network
class fcBaseline(base):
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
            with tf.variable_scope('fc_net'):
                w_fc1 = weight_variable((inputShape[0]*inputShape[1]*inputShape[2], self.params.num_fc1_units))
                b_fc1 = bias_variable((self.params.num_fc1_units,))
                fc1 = tf.nn.relu(tf.matmul(self.images, w_fc1) + b_fc1)

                w_fc2 = weight_variable((self.params.num_fc1_units, self.params.num_fc2_units))
                b_fc2 = bias_variable((self.params.num_fc2_units,))
                fc2 = tf.nn.relu(tf.matmul(fc1, w_fc2) + b_fc2)

            #Classification network
            with tf.variable_scope('classification_net'):
                output = fc2
                #Pass through classification network for reward
                w_logit = weight_variable((self.params.num_fc2_units, self.params.num_classes))
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
                var_list = tf.trainable_variables()
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
        feed_dict = {self.images: images, self.labels: labels}
        #Write flag
        #TODO see if you can put this into base
        if(step%self.params.write_step == 0):
            self.writeTrainSummary(feed_dict)
        #Run optimizers
        self.sess.run(self.train_op, feed_dict = feed_dict)

    def evalModelSummary(self, images, labels, injectAcc):
        feed_dict = {self.images: images, self.labels: labels,
                     self.injectBool: True, self.injectAcc:injectAcc}
        self.writeTestSummary(feed_dict)

    def evalModel(self, images, labels):
        feed_dict = {self.images: images, self.labels: labels}

        softmax_val = self.sess.run(self.softmax, feed_dict=feed_dict)

        #Find average of duplications
        #This is averaging across noisy locations
        #softmax_val = np.reshape(softmax_val, [self.params.M, -1, self.params.num_classes])
        #softmax_val = np.mean(softmax_val, 0)
        #Find label value

        pred_labels = np.argmax(softmax_val, 1).flatten()
        correct_count = np.sum(pred_labels == labels)
        return correct_count






