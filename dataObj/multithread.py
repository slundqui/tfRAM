import threading

class mtWrapper(object):
    def loadData(self, batchSize):
        self.loadBuf = self.dataObj.getData(batchSize)

    def __init__(self, dataObj, batchSize):
        self.dataObj = dataObj
        self.batchSize = batchSize
        #This is needed by tf code
        #self.num_train_examples = dataObj.num_train_examples

        #Start first thread
        self.loadThread = threading.Thread(target=self.loadData, args=(self.batchSize,))
        self.loadThread.start()

    #This function doesn't actually need numExample , but this api matches that of
    #image. So all we do here is assert numExample is the same
    def getData(self, numExample):
        assert(numExample == self.batchSize)
        #Block loadThread here
        self.loadThread.join()
        #Store loaded data into local variable
        #This should copy, not access reference
        returnBuf = self.loadBuf[:]
        #Launch new thread to load new buffer
        self.loadThread = threading.Thread(target=self.loadData, args=(self.batchSize,))
        self.loadThread.start()
        #Return stored buffer
        return returnBuf

    #Dont mt for test and val data
    def getTestData(self):
        return self.dataObj.getTestData()

    def getValData(self):
        return self.dataObj.getValData()
