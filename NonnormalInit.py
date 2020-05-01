'''Defines Keras initializer classes for non-normal RNN initializers'''

from keras.initializers import Initializer
import numpy as np

class ChainInit(Initializer):
    """Chain Initializer. Only use for 2D matrices.
    # Arguments
        diag_gain: Multiplicative factor to apply to the diagonal.
        offdiag_gain: Multiplicative factor to apply to the off-diagonal band.
    """
    def __init__(self, diag_gain=0.0, offdiag_gain=1.02):
        self.diag_gain = diag_gain
        self.offdiag_gain = offdiag_gain

    def __call__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError('Chain initializer can only be used for 2D matrices.')

        if shape[0] != shape[1]:
            raise ValueError('Dimensions must be identical.')

        return self.diag_gain * np.identity(shape[0]) + self.offdiag_gain * np.diag(np.ones(shape[0]-1), k=1)

    def get_config(self):
        return {'diag_gain': self.diag_gain, 'offdiag_gain': self.offdiag_gain}


class FbChainInit(Initializer):
    """Chain-with-feedback Initializer. Only use for 2D matrices.
    # Arguments
        diag_gain: Multiplicative factor to apply to the diagonal.
        offdiag_gain: Multiplicative factor to apply to the off-diagonal band.
    """
    def __init__(self, diag_gain=0.0, offdiag_gain=0.04):
        self.diag_gain = diag_gain
        self.offdiag_gain = offdiag_gain

    def __call__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError('Chain-with-feedback initializer can only be used for 2D matrices.')

        if shape[0] != shape[1]:
            raise ValueError('Dimensions must be identical.')

        return self.diag_gain * np.identity(shape[0]) + 0.99 * np.diag(np.ones(shape[0]-1), k=1) + \
               self.offdiag_gain * np.diag(np.ones(shape[0]-1), k=-1)

    def get_config(self):
        return {'diag_gain': self.diag_gain, 'offdiag_gain': self.offdiag_gain}


class NonnormalSourceInit(Initializer):
    """Non-normal source Initializer to be used as the input-to-hidden weight initializer in non-normal RNNs.
    # Arguments
        scale: scale factor to apply to the entire matrix.
    """
    def __init__(self, scale=0.9):
        self.scale = scale

    def __call__(self, shape, dtype=None):
        if len(shape) != 2:
            raise ValueError('Non-normal source initializer can only be used for 2D matrices.')

        return self.scale * np.eye(shape[0], shape[1], dtype=dtype)

    def get_config(self):
        return {'scale': self.scale}
