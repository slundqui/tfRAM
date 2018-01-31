import pdb
import numpy as np
import tensorflow as tf
from tf.dglimpse import GlimpseNet, LocNet

from tf.base import base
from tf.utils import weight_variable, bias_variable, loglikelihood, batchnorm

from plot.plot import plotGlimpseTrace
#import matplotlib.pyplot as plt

#Spatial transform network
class DRAM(base):

    def __init__(self, params):
        #Keeps track of locations
        self.loc_mean_arr = []
        self.sampled_loc_arr = []
        super(DRAM, self).__init__(params)

    def get_next_input(self, output, i):
        #Output is the output of the previous step in LSTM
        #i.e., hidden state h

        loc, loc_mean = self.loc_net(output)
        gl_next = self.gl(loc, self.is_train)

        self.loc_mean_arr.append(loc_mean)
        self.sampled_loc_arr.append(loc)
        return gl_next

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
                #self.addImageSummary('images', reshape_images, False)
                tf.summary.image('images', reshape_images)

                self.varDict['images'] = self.images

                self.rescale_images = self.images *2 - 1

                self.labels = tf.placeholder(tf.int64,
                        shape=[None],
                        name="labels")
                self.varDict['labels'] = self.labels

                self.is_train = tf.placeholder(tf.bool, shape=(), name="is_train")

            #Build aux nets
            # Build the aux nets.
            with tf.variable_scope('glimpse_net'):
                self.gl = GlimpseNet(self.params, self.rescale_images)
                #Add variables to varDict
                self.varDict.update(self.gl.getVars())

            with tf.variable_scope('loc_net'):
                self.loc_net = LocNet(self.params)
                #Add variables to varDict
                self.varDict.update(self.loc_net.getVars())

            #Build core network
            with tf.variable_scope('core_net'):
                # number of examples
                N = tf.shape(self.images)[0]
                #Initial glimpse location
                #TODO this should be calculated from context network
                init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
                #init_loc = tf.zeros((N, 2))

                #RNN base building block
                lstm_cell_0 = tf.contrib.rnn.BasicLSTMCell(self.params.cell_size_0)
                lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.params.cell_size_1)

                #Initial state
                state_0 = lstm_cell_0.zero_state(N, tf.float32)
                #TODO this should be initialized from context network
                state_1 = lstm_cell_1.zero_state(N, tf.float32)

                current_glimpse = self.gl(init_loc, self.is_train)
                outputs_0 = []
                outputs_1 = []
                glimpses = [current_glimpse]
                for i in range(self.params.num_glimpses):
                    output_0, state_0 = lstm_cell_0(current_glimpse, state_0)
                    #output_0 = batchnorm(output_0, "core_0", self.is_train)
                    outputs_0.append(output_0)

                    output_1, state_1 = lstm_cell_1(output_0, state_1)
                    #output_1 = batchnorm(output_1, "core_1", self.is_train)
                    outputs_1.append(output_1)

                    #Update glimpse
                    current_glimpse = self.get_next_input(output_1, i)
                    glimpses.append(current_glimpse)

                self.varDict["core_output_0"] = outputs_0
                self.varDict["core_output_1"] = outputs_1
                self.varDict["glimpses"] = glimpses

            #Baseline for variance reduction
            with tf.variable_scope('baseline'):
                baseline_w = weight_variable((self.params.cell_size_1, 1))
                baseline_b = bias_variable((1,))
                baselines = []
                for t, output in enumerate(outputs_1):
                    baseline_t = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
                    baseline_t = tf.squeeze(baseline_t)
                    baselines.append(baseline_t)
                baselines = tf.stack(baselines)  # [timesteps, batch_sz]
                baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

                self.varDict["baseline_w"] = baseline_w
                self.varDict["baseline_b"] = baseline_b
                self.varDict["baseline_h"] = baselines

            #Classification network
            with tf.variable_scope('classification_net'):
                # Take the last step of lower rnn only.
                output = outputs_0[-1]
                #Pass through classification network for reward
                classnet_w = weight_variable((self.params.cell_size_0, self.params.classnet_size))
                classnet_b = bias_variable((self.params.classnet_size,))
                logit_w = weight_variable((self.params.classnet_size, self.params.num_classes))
                logit_b = bias_variable((self.params.num_classes,))

                classnet_h = tf.nn.relu(tf.nn.xw_plus_b(output, classnet_w, classnet_b))
                logits = tf.nn.xw_plus_b(classnet_h, logit_w, logit_b)
                self.softmax = tf.nn.softmax(logits)

                self.varDict["classnet_w"] = classnet_w
                self.varDict["classnet_b"] = classnet_b
                self.varDict["classnet_h"] = classnet_h
                self.varDict["logit_w"] = logit_w
                self.varDict["logit_b"] = logit_b
                self.varDict["logits"] = logits

            with tf.variable_scope('loss'):
                # cross-entropy.
                xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels)
                xent = tf.reduce_mean(xent)
                pred_labels = tf.argmax(logits, 1)

                # 0/1 reward.
                reward = tf.cast(tf.equal(pred_labels, self.labels), tf.float32)
                rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
                rewards = tf.tile(rewards, (1, self.params.num_glimpses))  # [batch_sz, timesteps]

                logll = loglikelihood(self.loc_mean_arr, self.sampled_loc_arr, self.params.loc_std)
                #Only update rewards with reinforce
                advs = rewards - tf.stop_gradient(baselines)
                logllratio = tf.reduce_mean(logll * advs)
                reward = tf.reduce_mean(reward)

                #Baseline loss
                #Only update baselines with baseline_mse
                baselines_mse = tf.reduce_mean(tf.square((tf.stop_gradient(rewards) - baselines)))

                var_list = tf.trainable_variables()

                #hybrid loss
                #TODO baselines only update based on mse?
                hybrid_loss = -self.params.reinforce_lambda*logllratio + baselines_mse + xent

                grads = tf.gradients(hybrid_loss, var_list)
                grads, _ = tf.clip_by_global_norm(grads, self.params.max_grad_norm)

                #Add to scalar tensorboard
                self.scalarDict['hybrid_loss'] = hybrid_loss
                self.scalarDict['xent'] = xent
                self.scalarDict['baseline_mse'] = baselines_mse
                self.scalarDict['reinforce_loss'] = -logllratio
                self.scalarDict['reward'] = reward

                self.varDict['logll'] = logll
                self.varDict['loc_mean_arr'] = self.loc_mean_arr
                self.varDict['sampled_loc_arr'] = self.sampled_loc_arr

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
                #opt = tf.train.MomentumOptimizer(learning_rate, .9, use_nesterov=True)

                #Makes sure to update batchnorm var before training step
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.train_op = opt.apply_gradients(zip(grads, var_list), global_step=global_step)

                self.scalarDict['learning_rate'] = learning_rate

    def trainStep(self, step, dataObj):
        (images, labels) = dataObj.getData(self.params.batch_size)
        #Duplicate M times (see eqn 2)
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])
        #Build feeddict
        feed_dict = {self.images: images, self.labels: labels, self.is_train:True}
        #Write flag
        #TODO see if you can put this into base
        if(step%self.params.write_step == 0):
            self.writeTrainSummary(feed_dict)
        #Run optimizers
        self.sess.run(self.train_op, feed_dict = feed_dict)

    def evalModelSummary(self, images, labels, injectAcc):
        # Duplicate M times
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])

        feed_dict = {self.images: images, self.labels: labels,
                     self.injectBool: True, self.injectAcc:injectAcc,
                     self.is_train: False}
        self.writeTestSummary(feed_dict)

    def evalModel(self, images, labels):
        #labels_bak = labels
        # Duplicate M times
        # This repeats each experiment 10 times with random conditions
        labels_bak = labels
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])
        feed_dict = {self.images: images, self.labels: labels, self.is_train: False}

        softmax_val = self.sess.run(self.softmax, feed_dict=feed_dict)

        #Find average of duplications
        #This is averaging across noisy locations
        softmax_val = np.reshape(softmax_val, [self.params.M, -1, self.params.num_classes])
        softmax_val = np.mean(softmax_val, 0)
        #Find label value
        pred_labels = np.argmax(softmax_val, 1).flatten()
        correct_count = float(np.sum(pred_labels == labels_bak))
        return correct_count

    def plot(self, step, dataObj):
        self.plotGlimpse(step, dataObj)

    def plotGlimpse(self, step, dataObj, numPlot=5):
        test_imgs = dataObj.getTestData()[0][:numPlot, ...]
        feed_dict = {self.images: test_imgs, self.is_train:False}

        #Get glimpse locations
        (np_loc_mean_arr, np_sampled_loc_arr) = self.sess.run(
                [self.loc_mean_arr, self.sampled_loc_arr],
                feed_dict=feed_dict)

        #[nBatch, nGlimpse, 2] where 2 is y by x coords
        np_loc_mean_arr = np.transpose(np.array(np_loc_mean_arr), [1, 0, 2])
        np_sampled_loc_arr = np.transpose(np.array(np_sampled_loc_arr), [1, 0, 2])

        #Reshape into image
        reshape_test_imgs = np.reshape(test_imgs, (numPlot,) + dataObj.inputShape)

        outdir = self.plot_dir + "/" + str(step) + "/"
        self.makeDir(outdir)

        plotGlimpseTrace(reshape_test_imgs, np_loc_mean_arr, outdir, self.params.loc_pixel_ratio, nameprefix="mean")
        plotGlimpseTrace(reshape_test_imgs, np_sampled_loc_arr, outdir, self.params.loc_pixel_ratio, nameprefix="sampled")
