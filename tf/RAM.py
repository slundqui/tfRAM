import pdb
import numpy as np
import tensorflow as tf
from tf.glimpse import GlimpseNet, LocNet

from tf.base import base
from tf.utils import weight_variable, bias_variable, loglikelihood

from plot.plot import plotGlimpseTrace
#import matplotlib.pyplot as plt

#Spatial transform network
class RAM(base):

    def __init__(self, params):
        #Keeps track of locations
        self.loc_mean_arr = []
        self.sampled_loc_arr = []
        super(RAM, self).__init__(params)

    def get_next_input(self, output, i):
        #Output is the output of the previous step in LSTM
        #i.e., hidden state h
        loc, loc_mean = self.loc_net(output, self.eval_ph)
        gl_next = self.gl(loc)
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

                #self.rescale_images = self.images *2 - 1

                self.labels = tf.placeholder(tf.int64,
                        shape=[None],
                        name="labels")
                self.varDict['labels'] = self.labels

                self.eval_ph = tf.placeholder(tf.bool, shape=(), name="eval_ph")

            #Build aux nets
            # Build the aux nets.
            with tf.variable_scope('glimpse_net'):
                self.gl = GlimpseNet(self.params, self.images)
                #Add variables to varDict
                self.varDict.update(self.gl.getVars())

            with tf.variable_scope('loc_net'):
                self.loc_net = LocNet(self.params)
                #Add variables to varDict
                self.varDict.update(self.loc_net.getVars())

            # number of examples
            N = tf.shape(self.images)[0]
            #Initial glimpse location
            init_loc = tf.random_uniform((N, 2), minval=-1, maxval=1)
            #init_loc = tf.zeros((N, 2))

            init_glimpse = self.gl(init_loc)

            #Build core network
            with tf.variable_scope('core_net'):
                #Various base classes
                rnn_cell = tf.nn.rnn_cell
                seq2seq = tf.contrib.legacy_seq2seq

                #RNN base building block
                lstm_cell = rnn_cell.LSTMCell(self.params.cell_size, state_is_tuple=True)
                #Initial state
                init_state = lstm_cell.zero_state(N, tf.float32)
                inputs = [init_glimpse]
                inputs.extend([0] * (self.params.num_glimpses))
                #RNN
                outputs, _ = seq2seq.rnn_decoder(
                    inputs, init_state, lstm_cell, loop_function=self.get_next_input)

            #Baseline for variance reduction
            with tf.variable_scope('baseline'):
                w_baseline = weight_variable((self.params.cell_size, 1))
                b_baseline = bias_variable((1,))
                baselines = []
                for t, output in enumerate(outputs[1:]):
                    baseline_t = tf.nn.xw_plus_b(output, w_baseline, b_baseline)
                    baseline_t = tf.squeeze(baseline_t)
                    baselines.append(baseline_t)
                baselines = tf.stack(baselines)  # [timesteps, batch_sz]
                baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

            #Classification network
            with tf.variable_scope('classification_net'):
                # Take the last step only.
                output = outputs[-1]
                #Pass through classification network for reward
                w_logit = weight_variable((self.params.cell_size, self.params.num_classes))
                b_logit = bias_variable((self.params.num_classes,))
                logits = tf.nn.xw_plus_b(output, w_logit, b_logit)
                self.softmax = tf.nn.softmax(logits)

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
                advs = rewards - tf.stop_gradient(baselines)
                logllratio = tf.reduce_mean(logll * advs)
                reward = tf.reduce_mean(reward)

                #Baseline loss
                baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
                var_list = tf.trainable_variables()

                core_net_var = [v for v in var_list if "glimpse_net" in v.name or
                                                               "core_net" in v.name or
                                                               "baseline" in v.name]
                loc_net_var = [v for v in var_list if "loc_net" in v.name]

                assert(len(var_list) == len(core_net_var) + len(loc_net_var))

                #hybrid loss
                #TODO baselines only update based on mse?
                sup_loss = xent + baselines_mse # `-` for minimize
                reinforce_loss = -logllratio

                core_grads = tf.gradients(sup_loss, core_net_var)
                core_grads, _ = tf.clip_by_global_norm(core_grads, self.params.max_grad_norm)

                loc_grads = tf.gradients(reinforce_loss, loc_net_var)
                loc_grads, _ = tf.clip_by_global_norm(loc_grads, self.params.max_grad_norm)

                #Add to scalar tensorboard
                self.scalarDict['baselines_mse'] = baselines_mse
                self.scalarDict['xent'] = xent
                self.scalarDict['logllratio'] = logllratio
                self.scalarDict['reward'] = reward
                self.scalarDict['sup_loss'] = sup_loss
                self.scalarDict['reinforce_loss'] = reinforce_loss

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

                core_train_op = opt.apply_gradients(zip(core_grads, core_net_var), global_step=global_step)
                loc_train_op = opt.apply_gradients(zip(loc_grads, loc_net_var), global_step=global_step)
                self.train_op = tf.group(core_train_op, loc_train_op)

                self.scalarDict['learning_rate'] = learning_rate

    def trainStep(self, step, dataObj):
        (images, labels) = dataObj.getData(self.params.batch_size)
        #Duplicate M times (see eqn 2)
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])
        #Build feeddict
        feed_dict = {self.images: images, self.labels: labels, self.eval_ph:False}
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
                     self.eval_ph: self.params.det_eval}
        self.writeTestSummary(feed_dict)

    def evalModel(self, images, labels):
        #labels_bak = labels
        # Duplicate M times
        # This repeats each experiment 10 times with random conditions
        images = np.tile(images, [self.params.M, 1])
        labels = np.tile(labels, [self.params.M])
        feed_dict = {self.images: images, self.labels: labels, self.eval_ph: self.params.det_eval}

        softmax_val = self.sess.run(self.softmax, feed_dict=feed_dict)

        #Find average of duplications
        #This is averaging across noisy locations
        #softmax_val = np.reshape(softmax_val, [self.params.M, -1, self.params.num_classes])
        #softmax_val = np.mean(softmax_val, 0)
        #Find label value

        pred_labels = np.argmax(softmax_val, 1).flatten()
        correct_count = float(np.sum(pred_labels == labels))/self.params.M
        return correct_count

    def plot(self, step, dataObj):
        self.plotGlimpse(step, dataObj)

    def plotGlimpse(self, step, dataObj, numPlot=5):
        test_imgs = dataObj.getTestData()[0][:numPlot, ...]
        feed_dict = {self.images: test_imgs, self.eval_ph:self.params.det_eval}

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

        plotGlimpseTrace(reshape_test_imgs, np_loc_mean_arr, outdir, nameprefix="mean")
        plotGlimpseTrace(reshape_test_imgs, np_sampled_loc_arr, outdir, nameprefix="sampled")
