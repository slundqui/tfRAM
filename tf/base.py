import pdb
import numpy as np
import tensorflow as tf
import os
import scipy.sparse as sp
import subprocess
import json
import time
from tf.utils import createImageBuf, normImage
import inspect

class base(object):
    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params):
        #Global timestep
        self.timestep = 0
        #For storing model tensors
        self.imageDict = {}
        self.scalarDict = {}
        self.varDict = {}

        self.params = params
        self.tf_dir = self.params.run_dir + "/tfout/"
        self.ckpt_dir = self.params.run_dir + "/checkpoints/"
        self.save_file = self.ckpt_dir + "/save-model"
        self.plot_dir = self.params.run_dir + "/plots/"
        self.makeDirs()

        self.printParamsStr(params)
        self.printRepoDiffStr()

        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpuPercent)
        #config = tf.ConfigProto(gpu_options=gpu_options)

        #If device is set to cpu, set gpu cout to 0
        config = tf.ConfigProto()

        if(self.params.device[1:4] == "cpu"):
            config.device_count['GPU'] = 0

        config.gpu_options.allow_growth=True
        #config.gpu_options.per_process_gpu_memory_fraction=self.gpuPercent
        config.allow_soft_placement=True
        self.sess = tf.Session(config=config)

        self.buildModel()
        self.buildSummaries()
        self.initialize()
        self.writeSummary()

    def printRepoDiffStr(self):
        repolabel = subprocess.check_output(["git", "log", "-n1"])
        diffStr = subprocess.check_output(["git", "diff", "HEAD"])
        outstr = repolabel.decode() + "\n\n" + diffStr.decode()

        outfile = self.params.run_dir+"/repo.diff"

        #Will replace current file if found
        f = open(outfile, 'w')
        f.write(outstr)
        f.close()

    def genParamsStr(self, params):
        param_dict = {i:getattr(params, i) for i in dir(params) if not inspect.ismethod(i) and "__" not in i}
        paramsStr = json.dumps(param_dict, indent=2)
        return paramsStr

    def printParamsStr(self, params):
        outstr = self.genParamsStr(params)
        outfile = self.params.run_dir+"/params.json"
        #Will replace current file if found
        f = open(outfile, 'w')
        f.write(outstr)
        f.close()

    def makeDir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        self.makeDir(self.params.run_dir)
        self.makeDir(self.plot_dir)
        self.makeDir(self.ckpt_dir)

    def trainModel(self, trainDataObj):
        progress_time = time.time()
        for i in range(self.params.num_steps):
            if(i%self.params.eval_period == 0):
                #Evaluate test frame, providing gt so that it writes to summary
                self.evalModelBatch(trainDataObj)
                print("Done test eval")
            if(i%self.params.save_period == 0):
                #Save meta graph if beginning of run
                if(i == 0):
                    write_meta_graph = True
                else:
                    write_meta_graph = False
                save_path = self.saver.save(self.sess, self.save_file, global_step=self.timestep, write_meta_graph=write_meta_graph)
                print("Model saved in file: %s" % save_path)
            #Progress print
            if(i%self.params.progress == 0):
                tmp_time = time.time()
                print("Timestep ", self.timestep, ":", float(self.params.progress)/(tmp_time - progress_time), " iterations per second")
                progress_time = tmp_time
            if(i%self.params.plot_period == 0):
                self.plot(i, trainDataObj)

            self.trainStep(i, trainDataObj)

            self.timestep+=1

    def plot(self, step, dataObj):
        #Subclass should overwrite this
        pass

    def trainStep(self, step, trainDataObj):
        #Subclass must overwrite this
        assert(False)

    def buildModel(self):
        print("Cannot call base class buildModel")
        assert(0)

    def addImageSummary(self, name, tensor, normalize=True):
        assert(len(tensor.get_shape()) == 4)
        self.imageDict[name] = (tensor, normalize)

    def buildSummaries(self):
        ##Summaries ops

        #TODO fix this
        ##Write all images as a grid
        #with tf.name_scope("Summary"):
        #    opsList = []
        #    opsList_test = []
        #    gridList = {}
        #    gridList_test = {}

        #    for key in self.imageDict.keys():
        #        (tensor, normalize) = self.imageDict[key]
        #        (grid_image, grid_op) = createImageBuf(tensor, key+"_grid")
        #        (grid_image_test, grid_op_test) = createImageBuf(tensor, key+"_grid_test")

        #        gridList[key] = (grid_image, normalize)
        #        gridList_test[key] = (grid_image_test, normalize)

        #        opsList.append(grid_op)
        #        opsList_test.append(grid_op_test)
        #    if(len(opsList)):
        #        self.updateImgOp = tf.tuple(opsList)
        #    else:
        #        self.updateImgOp = tf.no_op()
        #    if(len(opsList_test)):
        #        self.updateImgOp_test = tf.tuple(opsList_test)
        #    else:
        #        self.updateImgOp_test = tf.no_op()

        trainSummaries = []
        testSummaries = []
        bothSummaries = []
        #for key in gridList.keys():
        for key in self.imageDict.keys():
            #(tensor, normalize) = gridList[key]
            #(test_tensor, test_normalize) = gridList_test[key]
            #assert(test_normalize == normalize)
            #Create images per item in imageDict
            #trainSummaries.append(tf.summary.image(key+"_grid_train", normImage(tensor, normalize)))
            #testSummaries.append(tf.summary.image(key+"_grid_test", normImage(test_tensor, test_normalize)))
            bothSummaries.append(tf.summary.image(key, normImage(self.imageDict[key][0], self.imageDict[key][1])))
            bothSummaries.append(tf.summary.histogram(key, self.imageDict[key][0]))

        #Output tensorboard summaries
        for key in self.scalarDict.keys():
            bothSummaries.append(tf.summary.scalar(key,  self.scalarDict[key]))

        #Generate histograms for all inputs in varDict
        for key in self.varDict.keys():
            bothSummaries.append(tf.summary.histogram(key, self.varDict[key]))

        #Merge summaries
        trainSummaries.extend(bothSummaries)
        testSummaries.extend(bothSummaries)
        self.mergeTrainSummary = tf.summary.merge(trainSummaries)
        self.mergeTestSummary = tf.summary.merge(testSummaries)

    def writeTrainSummary(self, feed_dict):
        trainSummary = self.sess.run(self.mergeTrainSummary, feed_dict=feed_dict)
        self.train_writer.add_summary(trainSummary, self.timestep)
        #Update image grid buffers
        #self.sess.run(self.updateImgOp, feed_dict=feed_dict)

    def writeTestSummary(self, feed_dict):
        testSummary = self.sess.run(self.mergeTestSummary, feed_dict=feed_dict)
        self.test_writer.add_summary(testSummary, self.timestep)
        #Update image grid buffers
        #self.sess.run(self.updateImgOp_test, feed_dict=feed_dict)

    def getLoadVars(self):
        return tf.global_variables()

    def initialize(self):
        with tf.name_scope("Savers"):
            ##Define saver
            load_v = self.getLoadVars()
            ##Load specific variables, save all variables
            self.loader = tf.train.Saver(var_list=load_v)
            self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=2)

            #Initialize
            self.initSess()
            #Load checkpoint if flag set
            if(self.params.load):
                self.loadModel()

    def guarantee_initialized_variables(self, session, list_of_variables = None):
        if list_of_variables is None:
            list_of_variables = tf.all_variables()
        uninitialized_variables = list(tf.get_variable(name) for name in
                session.run(tf.report_uninitialized_variables(list_of_variables)))
        session.run(tf.initialize_variables(uninitialized_variables))
        return uninitialized_variables

    #Initializes session.
    def initSess(self):
        self.sess.run(tf.global_variables_initializer())

    #Allocates and specifies the output directory for tensorboard summaries
    def writeSummary(self):
        self.mergedSummary = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.tf_dir + "/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tf_dir + "/test")

    def closeSess(self):
        self.sess.close()
        tf.reset_default_graph()

    def evalModel(self, images, labels):
        assert(False)

    def evalModelBatch(self, dataObj, writeOut = False):
        #Do validation
        (all_images, all_labels) = dataObj.getTestData()
        accuracy = self.evalSet(all_images, all_labels)
        if(writeOut):
            with open(self.params.run_dir + "accuracy.txt", "w") as f:
                f.write(str(accuracy))
        return accuracy

    def evalSet(self, allImages, allLabels):
        (numExamples, _) = allImages.shape
        assert(numExamples == allLabels.shape[0])
        #TODO remove this
        assert(numExamples % self.params.eval_batch_size == 0)
        steps_per_epoch = numExamples // self.params.eval_batch_size
        correct_count = 0.0
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
        return accuracy

    def evalModelSummary(self, images, labels, injectAcc):
        assert(False)

    #Loads a tf checkpoint
    def loadModel(self):
        self.loader.restore(self.sess, self.params.load_file)
        #self.guarantee_initialized_variables(self.sess)

        print("Model %s loaded" % self.params.load_file)





