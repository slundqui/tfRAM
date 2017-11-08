import pdb
import numpy as np
import tensorflow as tf
import os
import scipy.sparse as sp
import subprocess
import json
import time
from util import createImageBuf, normImage

class base(object):
    #Global timestep
    timestep = 0
    plotTimestep = 0

    #For storing model tensors
    imageDict = {}
    scalarDict = {}
    varDict = {}

    #Constructor takes inputShape, which is a 3 tuple (ny, nx, nf) based on the size of the image being fed in
    def __init__(self, params, inputShape):
        self.params = params
        self.makeDirs()
        #TODO
        #self.printParamsStr(params)
        #self.printRepoDiffStr()

        #Add node for printing params to tensorboard
        #TODO see if this can be done
        #tf.summary.text("params", tf.convert_to_tensor(paramsStr))

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

        self.buildModel(inputShape)
        self.buildSummaries()
        self.initialize()
        self.writeSummary()

    def printRepoDiffStr(self):
        repolabel = subprocess.check_output(["git", "log", "-n1"])
        diffStr = subprocess.check_output(["git", "diff", "HEAD"])
        outstr = repolabel + "\n\n" + diffStr

        outfile = self.params.run_dir+"/repo.diff"

        #Will replace current file if found
        f = open(outfile, 'w')
        f.write(outstr)
        f.close()

    def genParamsStr(self, params):
        pdb.set_trace()
        #TODO
        paramsStr = json.dumps(params, indent=2)
        return paramsStr

    def printParamsStr(self, params):
        outstr = self.genParamsStr(params)
        outfile = self.params.run_dir+"/params.json"
        #Will replace current file if found
        f = open(outfile, 'w')
        f.write(outstr)
        f.close()

    #Make approperiate directories if they don't exist
    def makeDirs(self):
        if not os.path.exists(self.params.run_dir):
            os.makedirs(self.params.run_dir)
        if not os.path.exists(self.params.plot_dir):
            os.makedirs(self.params.plotDir)
        if not os.path.exists(self.params.ckpt_dir):
            os.makedirs(self.params.ckpt_dir)

    def trainModel(self, trainDataObj, testDataObj=None):
        progress_time = time.time()
        for i in range(self.params.num_steps):
            if(i%self.params.eval_period == 0 and testDataObj is not None):
                #Evaluate test frame, providing gt so that it writes to summary
                data = testDataObj.getData(self.batch_size)
                self.evalModel(data[0], data[1], calcOutput=False)
                print "Done test eval"
            #Plot flag
            if(i%self.params.save_period == 0):
                #Save meta graph if beginning of run
                if(i == 0):
                    write_meta_graph = True
                else:
                    write_meta_graph = False
                save_path = self.saver.save(self.sess, self.params.save_file, global_step=self.timestep, write_meta_graph=write_meta_graph)
                print("Model saved in file: %s" % save_path)
            #Progress print
            if(i%self.progress == 0):
                tmp_time = time.time()
                print "Timestep ", self.timestep, ":", float(self.params.progress)/(tmp_time - progress_time), " iterations per second"
                progress_time = tmp_time

            self.trainStep(i, trainDataObj)

            self.timestep+=1

    def plotImage(self, step):
        #Subclass must overwrite this
        assert(False)

    def trainStep(self, step, trainDataObj):
        #Subclass must overwrite this
        assert(False)

    def buildModel(self, inputShape):
        print "Cannot call base class buildModel"
        assert(0)

    def addImageSummary(self, name, tensor, normalize=True):
        assert(len(tensor.get_shape()) == 4)
        self.imageDict[name] = (tensor, normalize)

    def buildSummaries(self):
        ##Summaries ops
        #Write all images as a grid
        with tf.name_scope("Summary"):
            opsList = []
            opsList_test = []
            gridList = {}
            gridList_test = {}

            for key in self.imageDict.keys():
                (tensor, normalize) = self.imageDict[key]
                (grid_image, grid_op) = createImageBuf(tensor, key+"_grid")
                (grid_image_test, grid_op_test) = createImageBuf(tensor, key+"_grid_test")

                gridList[key] = (grid_image, normalize)
                gridList_test[key] = (grid_image_test, normalize)

                opsList.append(grid_op)
                opsList_test.append(grid_op_test)
            self.updateImgOp = tf.tuple(opsList)
            self.updateImgOp_test = tf.tuple(opsList_test)

        trainSummaries = []
        testSummaries = []
        bothSummaries = []
        for key in gridList.keys():
            (tensor, normalize) = gridList[key]
            (test_tensor, test_normalize) = gridList_test[key]
            assert(test_normalize == normalize)
            #Create images per item in imageDict
            trainSummaries.append(tf.summary.image(key+"_grid_train", normImage(tensor, normalize)))
            testSummaries.append(tf.summary.image(key+"_grid_test", normImage(test_tensor, test_normalize)))
            bothSummaries.append(tf.summary.image(key, normImage(self.imageDict[key][0], normalize)))
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
        self.sess.run(self.updateImgOp, feed_dict=feed_dict)

    def writeTestSummary(self, feed_dict):
        testSummary = self.sess.run(self.mergeTestSummary, feed_dict=feed_dict)
        self.test_writer.add_summary(testSummary, self.timestep)
        #Update image grid buffers
        self.sess.run(self.updateImgOp_test, feed_dict=feed_dict)

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
            if(self.load):
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
        self.train_writer = tf.summary.FileWriter(self.tfDir + "/train", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.tfDir + "/test")

    def closeSess(self):
        self.sess.close()
        tf.reset_default_graph()

    def evalModelBatch(self, dataObj):
        assert(False)

    def evalModel(self, image, gt=None, calcOutput=True):
        assert(False)

    #Loads a tf checkpoint
    def loadModel(self):
        self.loader.restore(self.sess, self.params.load_file)
        #self.guarantee_initialized_variables(self.sess)

        print("Model %s loaded" % self.params.load_file)





