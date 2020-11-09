'''
Class to save in binary files information generated during the simulations.
I am not going to use a regular Logger because want to save large arrays, and in binary.
These binary files should be placed in solid-state drive (SSD) due to speed.
'''
import h5py
import os
import numpy as np

class BinaryFileLooger:

    def __init__(self, output_folder = './'):
        self.__version__ = "0.1.0"
        self.output_folder = output_folder
        #create folder if it does not exist
        os.makedirs(self.output_folder, exist_ok=True)

    def save_arrays(self, file_name, arrays, keys):
        '''
        Saves to disk.
        :param file_name: output file
        :param arrays:
        :param keys:
        :return:
        '''
        N = len(arrays)

        outputHDF5FilaName = os.path.join(self.output_folder, file_name)

        h5pyFile = h5py.File(outputHDF5FilaName, 'w')
        for i in range(N):
            #output the array
            npArray = arrays[i]
            thisKey = keys[i]
            h5pyFile[thisKey] = npArray #store in hdf5
        h5pyFile.close()


if __name__ == '__main__':
    output_folder = './testxx'
    logger = BinaryFileLooger(output_folder)
    file_name = 'tired.hdf5'
    arrays = (np.ones( (3,4)), np.ones( (2,3)))
    keys = ('sex','sox')
    logger.save_arrays(file_name, arrays, keys)
    #then read from Matlab with our companion function readArrayFromHDF5.m.