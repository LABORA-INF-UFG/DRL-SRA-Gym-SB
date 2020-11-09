'''
Utility functions
'''

import numpy as np

def compare_tensors(x,y):
    this_shape = x.shape
    x=x.flatten()
    y=y.flatten()
    e = x - y
    er = np.abs(np.real(e))
    ei = np.abs(np.imag(e))
    ir = np.argmax(er)
    ii = np.argmax(ei)
    return np.mean(er), np.mean(ei), er[ir], x[ir], y[ir], np.unravel_index(ir,this_shape,'F'), er[ii], x[ii], y[ii], np.unravel_index(ii,this_shape,'F')
