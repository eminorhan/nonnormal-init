'''Utility functions and classes'''

import warnings
import numpy as np
from keras.callbacks import Callback
from keras.initializers import Identity, Orthogonal, RandomNormal
from NonnormalInit import ChainInit, FbChainInit, NonnormalSourceInit

class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

def InitLoader(init, init_scale, hidden_units):
    if init=='chain':
        kernel_initializer = NonnormalSourceInit(scale=0.9)
        recurrent_initializer = ChainInit(diag_gain=0.0, offdiag_gain=init_scale)
    elif init=='fbchain':
        kernel_initializer = NonnormalSourceInit(scale=0.9)
        recurrent_initializer = FbChainInit(diag_gain=0.0, offdiag_gain=init_scale)
    elif init=='orthogonal':
        kernel_initializer = RandomNormal(stddev=0.9/np.sqrt(hidden_units))
        recurrent_initializer = Orthogonal(gain=init_scale)
    elif init=='identity':
        kernel_initializer = RandomNormal(stddev=0.9/np.sqrt(hidden_units))
        recurrent_initializer = Identity(gain=init_scale)

    return kernel_initializer, recurrent_initializer

