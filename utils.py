from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import functools
import random
import os, sys, cv2
import re
import pickle



def lazy_property(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class DigitDataset(object):

    def __init__ (self, root, h, w, labelNum, trainRatio=0.8, mode='train'):
        
        self.data, self.label = self.csvLoader("{}/{}.csv".format(root,mode), mode)
        if mode == 'train':
            cutIdx = int(self.data.shape[0]*trainRatio)
            self.trainData, self.trainLabel = self.data[:cutIdx], self.label[:cutIdx]
            self.valData, self.valLabel = self.data[cutIdx:], self.label[cutIdx:]
        self.h, self.w, self.labelNum = h, w, labelNum

    def csvLoader(self, fileName, mode):
        
        tmp = np.loadtxt(fileName, dtype=np.str, delimiter=",")
        if mode == 'train':
            data = tmp[1:, 1:].astype(np.float)
            data = data.reshape((-1, 28, 28))[:,:,:,np.newaxis]
            label = tmp[1:, 0].astype(np.int)
            return data, label
        elif mode == 'test':
            data = tmp[1:, :].astype(np.float)
            data = data.reshape((-1, 28, 28))[:,:,:,np.newaxis]
            return data, None
    
    def preprocess(self, img):

        if len(img.shape) == 2:
            temp=np.zeros((self.h,self.w,1))
            temp[:,:,0] = cv2.resize(img, (self.h,self.w))
        else:
            temp=np.zeros((self.h, self.w, img.shape[2]))
            for k in xrange(img.shape[2]):
                temp[:,:,k] = cv2.resize(img[:,:,k], (self.h,self.w))
        return temp.astype(np.float)

    def genMinibatch(self, batchSize, data, label=None):

        length = data.shape[0]
        idxs = np.array(random.sample(range(length), length)) if label is not None else np.arange(length)
        i = 0

        while True:
            if i+batchSize <= length:
                batchIdxs = idxs[i:i+batchSize]
                i += batchSize
            else:
                batchIdxs = idxs[i:]
                idxs = np.array(random.sample(idxs, length))
                batchIdxs = np.concatenate((batchIdxs, idxs[0:(i+batchSize-length)]))
                i = i+batchSize-length
                print('Another epoch done.')
            batchIdxs = batchIdxs.astype(np.int)
            for j, idx in enumerate(batchIdxs):
                if j == 0:
                    batchData = self.preprocess(data[idx, :])[np.newaxis, :]
                else:
                    batchData = np.concatenate((batchData, self.preprocess(data[idx, :])[np.newaxis, :]))
            
            if label is None: batchLabelOneHot=None
            else:
                batchLabelOneHot = np.zeros((batchSize, self.labelNum), dtype=np.float)
                batchLabelOneHot[np.arange(batchSize), label[batchIdxs]] = 1

            yield [batchData, batchLabelOneHot]

