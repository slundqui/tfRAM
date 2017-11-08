import tensorflow as tf
import numpy as np
import pdb

def createImageBuf(img, name, numBuf=25):
    with tf.name_scope("ImageGrid"):
        [nbatch, ny, nx, nf] = img.get_shape()
        imageBuf = tf.Variable(tf.zeros((numBuf, ny, nx, nf), dtype="float32"), trainable=False, name=name+"_buf")
        #Operator for removing a slice from the back and concatenating new image from front
        newBuf = imageBuf[:-int(nbatch)]
        concatBuf = tf.concat((img, newBuf), axis=0)
        updateBufOp = imageBuf.assign(concatBuf)
        #Create tablou of images
        numX = np.floor(np.sqrt(numBuf))
        numY = np.ceil(numBuf/numX)
        numX = int(numX)
        numY = int(numY)

        yList = []
        for y in range(numY):
            linIdxOffset = y * numX
            xList = [imageBuf[linIdxOffset + i][tf.newaxis, :, :, :] for i in range(numX)]
            yList.append(tf.concat(xList, axis=2))

        gridImg = tf.concat(yList, axis=1)

        return(gridImg, updateBufOp)

def batchnorm(input, namePrefix, varDict, updateList, inputBnOffset, inputBnScale):
    with tf.name_scope("BatchNorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        inputShape = input.get_shape()
        channels = inputShape[-1]
        if(inputBnOffset is None):
            #offset = tf.get_variable(namePrefix+"_bn_offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            offset = tf.Variable(tf.zeros([channels], dtype=tf.float32), name=namePrefix+"_bn_offset")
            if(varDict is not None):
                varDict[namePrefix+"_bn_offset"] = offset
            if(updateList is not None):
                updateList.append(offset)
        else:
            offset = inputBnOffset

        if(inputBnScale is None):
            #scale = tf.get_variable(namePrefix+"_bn_scale", [channels], dtype=tf.float32, initializer=tf.ones_initializer())
            scale = tf.Variable(tf.ones([channels], dtype=tf.float32), name=namePrefix+"_bn_scale")
            if(varDict is not None):
                varDict[namePrefix+"_bn_scale"] = scale
            if(updateList is not None):
                updateList.append(scale)
        else:
            scale = inputBnScale

        if(len(inputShape) == 4):
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        elif(len(inputShape) == 2):
            mean, variance = tf.nn.moments(input, axes=[0], keep_dims=False)
        else:
            print "Unknown input shape"
            pdb.set_trace()
            assert(False)
        variance_epsilon = 2e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized

