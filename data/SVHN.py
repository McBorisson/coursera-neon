# import some useful packages
from neon.data import NervanaDataIterator
import numpy as np
import cPickle
import os

class SVHN(NervanaDataIterator):

    def __init__(self, X, Y, lshape):

        # Load the numpy data into some variables. We divide the image by 255 to normalize the values
        # between 0 and 1.
        self.X = X / 255.
        self.Y = Y
        self.shape = lshape  # shape of the input data (e.g. for images, (C, H, W))

        # 1. assign some required and useful attributes
        self.start = 0  # start at zero
        self.ndata = self.X.shape[0]  # number of images in X (hint: use X.shape)
        self.nfeatures = self.X.shape[1]  # number of features in X (hint: use X.shape)

        # number of minibatches per epoch
        # to calculate this, use the batchsize, which is stored in self.be.bsz
        self.nbatches = self.ndata/self.be.bsz 
        
        
        # 2. allocate memory on the GPU for a minibatch's worth of data.
        # (e.g. use `self.be` to access the backend.). See the backend documentation.
        # to get the minibatch size, use self.be.bsz
        # hint: X should have shape (# features, mini-batch size)
        # hint: use some of the attributes previously defined above
        self.dev_X = self.be.zeros((self.nfeatures, self.be.bsz))
        self.dev_Y = self.be.zeros((self.Y.shape[1], self.be.bsz))


    def reset(self):
        self.start = 0

    def __iter__(self):
        # 3. loop through minibatches in the dataset
        for index in range(self.start, self.ndata, self.be.bsz):
            # 3a. grab the right slice from the numpy arrays
            inputs = self.X[index:(index + self.be.bsz), :]
            targets = self.Y[index:(index + self.be.bsz), :]
            
            # The arrays X and Y data are in shape (batch_size, num_features),
            # but the iterator needs to return data with shape (num_features, batch_size).
            # here we transpose the data, and then store it as a contiguous array. 
            # numpy arrays need to be contiguous before being loaded onto the GPU.
            inputs = np.ascontiguousarray(inputs.T)
            targets = np.ascontiguousarray(targets.T)
                        
            # here we test your implementation
            # your slice has to have the same shape as the GPU tensors you allocated
            assert inputs.shape == self.dev_X.shape, \
                   "inputs has shape {}, but dev_X is {}".format(inputs.shape, self.dev_X.shape)
            assert targets.shape == self.dev_Y.shape, \
                   "targets has shape {}, but dev_Y is {}".format(targets.shape, self.dev_Y.shape)
            
            # 3b. transfer from numpy arrays to device
            # - use the GPU memory buffers allocated previously,
            #    and call the myTensorBuffer.set() function. 
            self.dev_X.set(inputs)
            self.dev_Y.set(targets)
            
            # 3c. yield a tuple of the device tensors.
            # X should be of shape (num_features, batch_size)
            # Y should be of shape (4, batch_size)
            yield (self.dev_X, self.dev_Y)