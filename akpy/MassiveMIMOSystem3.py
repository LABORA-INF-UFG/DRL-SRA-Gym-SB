'''
Massive MIMO system.

Each system corresponds to a fixed access scenario. If we change positions, we need to
create new object.
An episode is composed of several coherence blocks. This class deals at the coherence block
level, not episodes. The coherence block is composed by nbrOfRealizations TTIs.
A coherence block is created and all nbrOfRealizations channels H and H_hat created.
Now, for each TTI the external (RL) caller says what users are going to be served for
current TTI, and the code returns their SEs. For convenience, it basically puts zeros
on the channels of the non-served users.

    In case of MR, because the precoders are the estimated channels, using only subsets of users does not change
    the precoders and, consequently the signal and interferences. But the RZF and MMSE involve inverting matrices
    and using a subset of users changes the precoders. A test such as below confirms it:
    H=magic(3)
    H*inv(H'*H+eye(3))

    H=H(1:2,1:2);
    H*inv(H'*H+eye(2))

    Then, for RZF and MMSE methods, I put zeros before calculating the precoders.

I am giving the capability of dealing with up to 3 frequencies.

AK-TODO: from version 3, need to better split pilot time and transmission time
----------------------------- coherence block ---------------------------
pilot 1 - pilot 2 - ... - pilot taup  | TTI 1 - TTI 2 - ... - TTI num_tti
*** all used for channel estimation***|
H 1     - H 2     - ... - H taup      |

For each episode:
- I will assume the positions are fixed now (later I can generate new positions, i.e., use new .mat files)
- choose reasonable number of observations and calculate the matrices to generate channel estimates
  For each coherence block:
  * generate num_tti + 1 channels H
  * convert the first channel into Hhat, which will be used in the whole block
    For each TTI:
    # use H[2:] and Hhat to estimate SE, etc.

Based on:
%Emil Bjornson, Jakob Hoydis and Luca Sanguinetti (2017),
%"Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency",
%Foundations and Trends in Signal Processing: Vol. 11, No. 3-4,
%pp. 154-655. DOI: 10.1561/2000000093.
%https://www.massivemimobook.com
'''
#In this code, j is the BS (cell) of interest and l the interfering BS,
#k is the UE in the BS of interest, and i is the UE in the interfering BS (see e.g. (7.3))
#in all reshapes I need to use 'F': h_temp = np.reshape(h_temp,(self.K,self.L),'F')

from akpy.matlab_tofrom_python import read_matlab_array_from_mat
#from .matlab_tofrom_python import read_matlab_array_from_mat
import numpy as np
from scipy.linalg import sqrtm
#import matplotlib.pyplot as plt
import os
import copy
import sys
from akpy.util import compare_tensors

class MassiveMIMOSystem:
    def __init__(self, K=3, Kmax=2, frequency_index=1, num_bs_antennas=64):
        #channel estimation method: MMSE or
        self.channel_estimation_method = 'MMSE'
        #precoding and combining methods: MMSE, MR or RZF
        self.precoding_method = 'MR'
        #number of connected UEs. Cannot be larger than num_UEs_per_BS
        #to simplify, all cells have same number of UEs. But if
        #num_connected_users < num_UEs_per_BS, only the first elements
        #are valid for target_BS.
        #Number of UEs per BS
        self.num_UEs_per_BS = K #old K
        #below needs to be equals for now due to matrix dimensions
        self.num_connected_users_in_target_BS = self.num_UEs_per_BS
        #maximum number of served users
        self.Kmax = Kmax
        #Number of BSs
        self.L = 9
        #index of target BS. Only one is studied
        self.target_BS = 5-1 #starts from index 0 and 5 is the center when L=9
        #Number of BS antennas
        self.num_bs_antennas = num_bs_antennas #old M
        #Define the pilot reuse factor
        self.pilot_reuse_factor = 2 #old f
        #Select the number of setups with random UE locations
        #nbrOfSetups = 100
        #Select the number of channel realizations per setup
        self.nbrOfRealizations = 10
        ## Propagation parameters
        #Communication bandwidth
        B = 20e6

        #Total uplink transmit power per UE (mW)
        self.p = 100

        #Total downlink transmit power per UE (mW)
        rho = 100

        #Maximum downlink transmit power per BS (mW)
        self.Pmax = self.num_UEs_per_BS * rho

        #Compute downlink power per UE in case of equal power allocation
        self.rhoEqual = (self.Pmax / self.num_UEs_per_BS) * np.ones((self.num_UEs_per_BS, self.L))

        #Define noise figure at BS (in dB)
        noiseFigure = 7

        #Compute noise power
        noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

        #Select length of coherence block
        self.tau_c = 200

        #I am using ak_functionExampleSetup_resource_allocation.m
        #to generate R and channelGaindB
        #Use the approximation of the Gaussian local scattering model
        #accuracy = 2
        #Angular standard deviation in the local scattering model (in degrees)
        #ASDdeg = 10
        if os.name == 'nt':
            fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/R_channelGaindB' + str(frequency_index) + '.mat'
        else:
            fileName = '/mnt/d/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/R_channelGaindB' + str(frequency_index) + '.mat'
        channelGaindB=read_matlab_array_from_mat(fileName, 'channelGaindB')
        R=read_matlab_array_from_mat(fileName, 'R') #normalized to have norm = M

        this_num_bs_antennas = R.shape[0]
        if this_num_bs_antennas < self.num_bs_antennas:
            raise Exception('R is too small in ' + fileName)
        if this_num_bs_antennas > self.num_bs_antennas:
            #reduce matrix, self.R[:,:,k,j,l]
            R2 = np.zeros((self.num_bs_antennas,self.num_bs_antennas,self.num_UEs_per_BS,self.L,self.L), dtype=np.complex64)
            R2 = R[0:self.num_bs_antennas,0:self.num_bs_antennas,:,:,:]
            R = R2
            R2 = None

            #print('R', self.R.shape)

        #Compute the normalized average channel gain, where the normalization
        #is based on the noise power
        #noteh that the noise power sigma^2 will become 1 in the code (when compared to book) due to this normalization
        channelGainOverNoise = channelGaindB - noiseVariancedBm

        self.channel_gain_over_noise_linear = 10**(channelGainOverNoise/10)
        self.R_scaled = self.scale_correlation_matrix(R, channelGaindB)

        #from functionChannelEstimates.m
        #Length of pilot sequences
        self.tau_p = self.pilot_reuse_factor * self.num_UEs_per_BS

        #Generate pilot pattern
        if self.pilot_reuse_factor == 1:
            self.pilotPattern = np.ones((self.L,))
        elif self.pilot_reuse_factor == 2: #Only works in the running example with its 16 BSs
            #self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1]).flatten()
            if self.L == 16:
                self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1])
            elif self.L == 9:
                self.pilotPattern = np.array([[1,2,1,2,1,2,1,2,1]])
        elif self.pilot_reuse_factor == 4: #Only works in the running example with its 16 BSs
            #self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4]).flatten()
            self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4])
        elif self.pilot_reuse_factor == 16: #Only works in the running example with its 16 BSs
            self.pilotPattern = np.arange(1,self.L)

        #print('Pilot groups=', self.pilotPattern)
        #Store identity matrix of size M x M
        self.eyeM = np.eye(self.num_bs_antennas)

    def set_num_realizations(self, nbrOfRealizations):
        self.nbrOfRealizations = nbrOfRealizations

    '''
    Get self.nbrOfRealizations realizations of all channels.
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector H(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.    
    AK-TODO: these channels do not have time correlation. Could impose by using e.g. a moving average
    FIR filter.
    '''
    def channel_realizations(self):
        #Go through all channels and apply the channel gains to the spatial
        H = np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L) + \
            1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L)

        for j in range(self.L):
            for l in range(self.L):
                for k in range(self.num_UEs_per_BS):
                    #Rtemp = self.channel_gain_over_noise_linear[k,j,l] * self.R[:,:,k,j,l]
                    #print(Rtemp.shape)
                    #Rtemp=np.matrix([[8,1+3j,6],[3-100j,5,7],[4,9,2]])
                    Rsqrt= sqrtm(self.R_scaled[:,:,k,j,l])
                    #print(Rsqrt)
                    #exit(-1)
                    Htemp = H[:,:,k,j,l]
                    #Apply correlation to the uncorrelated channel realizations
                    H[:,:,k,j,l] = np.sqrt(0.5) * np.matmul(Rsqrt,Htemp)
        return H

    def scale_correlation_matrix(self,R,channelGaindB):
        #Go through all channels and apply the channel gains to the spatial
        #correlation matrices
        R_scaled = np.zeros(R.shape, dtype=np.complex64)
        for j in range(self.L):
            for l in range(self.L):
                for k in range(self.num_UEs_per_BS):
                    if channelGaindB[k,j,l]>-np.Inf:
                        #Extract channel gain in linear scale
                        #channel_gain_linear = 10**(self.channelGaindB[k,j,l]/10)
                        #Apply channel gain to correlation matrix
                        R_scaled[:,:,k,j,l] = R[:,:,k,j,l] * self.channel_gain_over_noise_linear[k,j,l]
                        #R_scaled[:,:,k,j,l] = R[:,:,k,j,l] * channel_gain_linear
                    else:
                        R_scaled[:,:,k,j,l] = 0
                        #H[:,:,k,j,l] = 0
        return R_scaled

    '''
    Linear MMSE channel estimator. From function ak_functionChannelEstimates.m
    Need to implement the EW-MMSE estimator and the LS estimator.
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector Hhat(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.    
    '''
    def channel_estimates_mmse(self,H):
        #Generate realizations of normalized noise
        Np = np.sqrt(0.5)*(np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor) +
                           1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor))
        #Prepare to store MMSE channel estimates
        Hhat_MMSE = np.zeros((self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)
        #Prepare to store estimation error correlation matrices
        C_MMSE = np.zeros((self.num_bs_antennas, self.num_bs_antennas, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)

        # Go through all cells
        for j in range(self.L):
            #Go through all f pilot groups
            for g in range(self.pilot_reuse_factor):
                #Extract the cells that belong to pilot group g
                groupMembers = np.where(self.pilotPattern == g+1) #add 1 because first pilot group is 1
                groupMembers = groupMembers[1] #AK-TODO buggy, fix it, now need to discard 1st dimension
                #if not groupMembers.any():
                #    raise RuntimeError('No cell is assigned to pilot group = ', g+1)
                #Compute processed pilot signal for all UEs that use these pilots, according to (3.5)

                Htemp = H[:,:,:,groupMembers,j]
                yp = np.sqrt(self.p)*self.tau_p*np.sum(Htemp,3) + np.sqrt(self.tau_p)*Np[:,:,:,j,g]
                #Go through all UEs
                for k in range(self.num_UEs_per_BS):
                    #Compute the matrix that is inverted in the MMSE estimator
                    Rtemp = self.R_scaled[:,:,k,groupMembers,j]
                    PsiInv = (self.p*self.tau_p*np.sum(Rtemp,2) + self.eyeM)
                    for l in groupMembers:
                        #Compute MMSE estimate of channel between BS l and UE k in
                        #cell j using (3.9) in Theorem 3.1
                        #x = B/A is the solution to the equation xA = B. mrdivide in Matlab
                        RPsi = np.matmul(self.R_scaled[:,:,k,l,j], np.linalg.inv(PsiInv))
                        Hhat_MMSE[:,:,k,l,j] = np.sqrt(self.p)*np.matmul(RPsi,yp[:,:,k])
                        #Compute corresponding estimation error correlation matrix, using (3.11)
                        C_MMSE[:,:,k,l,j] = self.R_scaled[:,:,k,l,j] - self.p*self.tau_p*np.matmul(RPsi,self.R_scaled[:,:,k,l,j])
        return Hhat_MMSE, C_MMSE

    '''
    Linear MMSE channel estimator. From function ak_functionChannelEstimates.m
    Need to implement the EW-MMSE estimator and the LS estimator.
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector Hhat(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.    
    '''
    def initialize_mmse_channel_estimator(self,H):
        #Generate realizations of normalized noise
        Np = np.sqrt(0.5)*(np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor) +
                           1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor))
        #Prepare to store MMSE channel estimates
        Hhat_MMSE = np.zeros((self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)
        #Prepare to store estimation error correlation matrices
        self.C_MMSE = np.zeros((self.num_bs_antennas, self.num_bs_antennas, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)
        self.RPsi = np.zeros((self.num_UEs_per_BS, self.L, self.L, self.num_bs_antennas, self.num_bs_antennas), dtype=np.complex64)

        # Go through all cells
        for j in range(self.L):
            #Go through all f pilot groups
            for g in range(self.pilot_reuse_factor):
                #Extract the cells that belong to pilot group g
                groupMembers = np.where(self.pilotPattern == g+1) #add 1 because first pilot group is 1
                groupMembers = groupMembers[1] #AK-TODO buggy, fix it, now need to discard 1st dimension
                #if not groupMembers.any():
                #    raise RuntimeError('No cell is assigned to pilot group = ', g+1)
                #Compute processed pilot signal for all UEs that use these pilots, according to (3.5)

                Htemp = H[:,:,:,groupMembers,j]
                yp = np.sqrt(self.p)*self.tau_p*np.sum(Htemp,3) + np.sqrt(self.tau_p)*Np[:,:,:,j,g]
                #Go through all UEs
                for k in range(self.num_UEs_per_BS):
                    #Compute the matrix that is inverted in the MMSE estimator
                    Rtemp = self.R_scaled[:,:,k,groupMembers,j]
                    PsiInv = (self.p*self.tau_p*np.sum(Rtemp,2) + self.eyeM)
                    for l in groupMembers:
                        #Compute MMSE estimate of channel between BS l and UE k in
                        #cell j using (3.9) in Theorem 3.1
                        #x = B/A is the solution to the equation xA = B. mrdivide in Matlab
                        self.RPsi[k,l,j] = np.sqrt(self.p)*np.matmul(self.R_scaled[:,:,k,l,j], np.linalg.inv(PsiInv))
                        Hhat_MMSE[:,:,k,l,j] = np.matmul(self.RPsi[k,l,j],yp[:,:,k])
                        #Compute corresponding estimation error correlation matrix, using (3.11)
                        #self.C_MMSE[:,:,k,l,j] = self.R_scaled[:,:,k,l,j] - self.p*self.tau_p*np.matmul(self.RPsi[k,l,j],self.R_scaled[:,:,k,l,j])
                        #note that np.sqrt(self.p) has been incorporated to the matrix, so use only another np.sqrt(self.p) below
                        self.C_MMSE[:,:,k,l,j] = self.R_scaled[:,:,k,l,j] - np.sqrt(self.p)*self.tau_p*np.matmul(self.RPsi[k,l,j],self.R_scaled[:,:,k,l,j])
        #return Hhat_MMSE, self.C_MMSE


    '''
    Linear MMSE channel estimator. From function ak_functionChannelEstimates.m
    Need to implement the EW-MMSE estimator and the LS estimator.
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector H(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.    
    This version does not compute C_MMSE
    '''
    def channel_estimates_mmse_version_without_covariance(self,H):
        #Generate realizations of normalized noise
        Np = np.sqrt(0.5)*(np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor) +
                           1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor))
        #Prepare to store MMSE channel estimates
        Hhat_MMSE = np.zeros((self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)
        #Prepare to store estimation error correlation matrices
        #C_MMSE = np.zeros((self.num_bs_antennas, self.num_bs_antennas, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)

        # Go through all cells
        for j in range(self.L):
            #Go through all f pilot groups
            for g in range(self.pilot_reuse_factor):
                #Extract the cells that belong to pilot group g
                groupMembers = np.where(self.pilotPattern == g+1) #add 1 because first pilot group is 1
                groupMembers = groupMembers[1] #AK-TODO buggy, fix it, now need to discard 1st dimension
                #if not groupMembers.any():
                #    raise RuntimeError('No cell is assigned to pilot group = ', g+1)
                #Compute processed pilot signal for all UEs that use these pilots, according to (3.5)

                Htemp = H[:,:,:,groupMembers,j]
                yp = np.sqrt(self.p)*self.tau_p*np.sum(Htemp,3) + np.sqrt(self.tau_p)*Np[:,:,:,j,g]
                #Go through all UEs
                for k in range(self.num_UEs_per_BS):
                    #Compute the matrix that is inverted in the MMSE estimator
                    Rtemp = self.R_scaled[:,:,k,groupMembers,j]
                    PsiInv = (self.p*self.tau_p*np.sum(Rtemp,2) + self.eyeM)
                    for l in groupMembers:
                        #Compute MMSE estimate of channel between BS l and UE k in
                        #cell j using (3.9) in Theorem 3.1
                        #x = B/A is the solution to the equation xA = B. mrdivide in Matlab
                        RPsi = np.matmul(self.R_scaled[:,:,k,l,j], np.linalg.inv(PsiInv))
                        Hhat_MMSE[:,:,k,l,j] = np.sqrt(self.p)*np.matmul(RPsi,yp[:,:,k])
                        #Compute corresponding estimation error correlation matrix, using (3.11)
                        #C_MMSE[:,:,k,l,j] = self.R_scaled[:,:,k,l,j] - self.p*self.tau_p*np.matmul(RPsi,self.R_scaled[:,:,k,l,j])
        return Hhat_MMSE #, C_MMSE


    '''
    from functionComputeSINR_DL.m
    Implements only the MMSE precoding (from MMSE combining and duality)
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector H(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.    
    signal_MMSE    = K x L matrix where element (k,j) is a_jk in (7.2)
    interf_MMSE    = K x L x K x L matrix where this is a permutation of L x K x L x K
                     that represents (l,i,j,k) in b_lijk in (7.3). In other words, the code uses an array that does not
                     exactly matches the book. 
    '''
    def compute_SINR_DL_mmse_precoding(self, H, Hhat, C, realizations_range):
        #Store identity matrices of different sizes
        #eyeK = np.eye(self.K)
        eyeM = np.eye(self.num_bs_antennas)

        #Compute sum of all estimation error correlation matrices at every BS
        C_totM = np.reshape(self.p * np.sum(np.sum(C,2),2), [self.num_bs_antennas, self.num_bs_antennas, self.L], 'F')

        #Compute the prelog factor assuming only downlink transmission
        #prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)

        #Prepare to store simulation results for signal gains
        signal_MMMSE = np.zeros((self.num_UEs_per_BS, self.L), dtype=np.complex64)

        #Prepare to store simulation results for Bernoulli_extra_interference powers
        #interf_MMMSE = np.zeros((self.K,self.L,self.K,self.L),dtype=np.complex64)
        interf_MMMSE = np.zeros((self.num_UEs_per_BS, self.L, self.num_UEs_per_BS, self.L)) #real-valued, not complex

        #AK-TODO: make a smarter implementation
        number_of_realizations = 0
        for n in realizations_range: #range(self.nbrOfRealizations): #Go through all channel realizations
            number_of_realizations += 1

        for n in realizations_range: #range(self.nbrOfRealizations): #Go through all channel realizations
            #Go through all cells
            if n < 0 or n > self.nbrOfRealizations:
                raise Exception(str(n) + ' is outside the range [0, ' + str(self.nbrOfRealizations) + ']')
            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j
                Hallj = np.reshape(H[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Extract channel realizations from all UEs to BS j
                #Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')
                #note that we use n=0 as the estimate in the beginning of the coherence block, which will be used throughout the block
                Hhatallj = np.reshape(Hhat[:,0,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Compute MR combining in (4.11)
                V_MR = Hhatallj[:, self.num_UEs_per_BS * j:self.num_UEs_per_BS * (j + 1)]
                #print('AK', V_MR)

                #Compute M-MMSE combining in (4.7)
                #Backslash or matrix left division. If A is a square matrix, A\B is roughly the same as inv(A)*B, except it is computed in a different way
                tempM = self.p*(self.p*(np.matmul(Hhatallj,np.conj(np.transpose(Hhatallj)))) + C_totM[:,:,j]+eyeM)
                V_MMMSE = np.matmul(np.linalg.inv(tempM), V_MR)
                #Go through all UEs in cell j
                for k in range(self.num_UEs_per_BS):
                    if np.linalg.norm(V_MR[:,k])>0:
                        #M-MMSE precoding
                        w = V_MMMSE[:,k]/np.linalg.norm(V_MMMSE[:,k]) #Extract precoding vector
                        w = np.reshape(w, (1,self.num_bs_antennas)).conj() #Hermitian: make it a row vector and conjugate

                        #Compute realizations of the terms inside the expectations
                        #of the signal and Bernoulli_extra_interference terms of (7.2) and (7.3)
                        h_temp = H[:,n,k,j,j]
                        signal_MMMSE[k,j] = signal_MMMSE[k,j] + (np.inner(w,h_temp))/number_of_realizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp, (self.num_UEs_per_BS, self.L), 'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_MMMSE[k,j,:,:] = interf_MMMSE[k,j,:,:] + h_temp/number_of_realizations
        #Compute the terms in (7.2)
        signal_MMMSE = np.abs(signal_MMMSE)**2
        #print('AK2', signal_MMMSE)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.num_UEs_per_BS):
                interf_MMMSE[k,j,k,j] = interf_MMMSE[k,j,k,j] - signal_MMMSE[k,j]
        return signal_MMMSE[:,self.target_BS], interf_MMMSE

    '''
    from functionComputeSINR_DL.m
    Implements only the MR precoding (from MR combining and duality)
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector H(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j and the BS in cell l.    
    signal_MR    = K x L matrix where element (k,j) is a_jk in (7.2)
    interf_MR    = K x L x K x L matrix where this is a permutation of L x K x L x K
                     that represents (l,i,j,k) in b_lijk in (7.3). In other words, the code uses an array that does not
                     exactly matches the book.
    '''
    def compute_SINR_DL_mr_precoding(self, H, Hhat, realizations_range):
        #Prepare to store simulation results for signal gains
        signal_MR = np.zeros((self.num_UEs_per_BS, self.L), dtype=np.complex64)

        #Prepare to store simulation results for Bernoulli_extra_interference powers
        #interf_MMMSE = np.zeros((self.K,self.L,self.K,self.L),dtype=np.complex64)
        interf_MR = np.zeros((self.num_UEs_per_BS, self.L, self.num_UEs_per_BS, self.L)) #real-valued, not complex

        #AK-TODO: make a smarter implementation
        number_of_realizations = 0
        for n in realizations_range: #range(self.nbrOfRealizations): #Go through all channel realizations
            number_of_realizations += 1

        for n in realizations_range: #range(self.nbrOfRealizations): #Go through all channel realizations
            #Go through all cells
            if n < 0 or n > self.nbrOfRealizations:
                raise Exception(str(n) + ' is outside the range [0, ' + str(self.nbrOfRealizations) + ']')

            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j, which includes all UEs in cells different than j
                Hallj = np.reshape(H[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Extract channel realizations from all UEs to BS j
                #Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')
                #note that we use n=0 as the estimate in the beginning of the coherence block, which will be used throughout the block
                Hhatallj = np.reshape(Hhat[:,0,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Compute MR combining in (4.11)
                #simply extract V_MR corresponding to cell j from Hhat
                V_MR = Hhatallj[:, self.num_UEs_per_BS * j:self.num_UEs_per_BS * (j + 1)]
                #print('AK', V_MR)

                #Compute M-MMSE combining in (4.7)
                #Backslash or matrix left division. If A is a square matrix, A\B is roughly the same as inv(A)*B, except it is computed in a different way
                #tempM = self.p*(self.p*(np.matmul(Hhatallj,np.conj(np.transpose(Hhatallj)))) + C_totM[:,:,j]+eyeM)
                #V_MMMSE = np.matmul(np.linalg.inv(tempM), V_MR)
                #Go through all UEs in cell j
                for k in range(self.num_UEs_per_BS):
                    if np.linalg.norm(V_MR[:,k])>0:
                        #MR precoding
                        w = V_MR[:,k]/np.linalg.norm(V_MR[:,k]) #Extract precoding vector
                        w = np.reshape(w, (1,self.num_bs_antennas)).conj() #Hermitian: make it a row vector and conjugate

                        #Compute realizations of the terms inside the expectations
                        #of the signal and Bernoulli_extra_interference terms of (7.2) and (7.3)
                        h_temp = H[:,n,k,j,j]
                        signal_MR[k,j] = signal_MR[k,j] + (np.inner(w,h_temp))/number_of_realizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp, (self.num_UEs_per_BS, self.L), 'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_MR[k,j,:,:] = interf_MR[k,j,:,:] + h_temp/number_of_realizations
        #Compute the terms in (7.2)
        signal_MR = np.abs(signal_MR)**2
        #print('AK2', signal_MR)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.num_UEs_per_BS):
                interf_MR[k,j,k,j] = interf_MR[k,j,k,j] - signal_MR[k,j]

        #return only the signal for the target BS but all Bernoulli_extra_interference
        return signal_MR[:,self.target_BS], interf_MR


    '''
    from functionComputeSINR_DL.m
    Implements only the RZF precoding (from RZF combining and duality)
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L. 
    The vector H(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.    
    signal_RZF    = K x L matrix where element (k,j) is a_jk in (7.2)
    interf_RZF    = K x L x K x L matrix where this is a permutation of L x K x L x K
                     that represents (l,i,j,k) in b_lijk in (7.3). In other words, the code uses an array that does not
                     exactly matches the book.
    '''
    def compute_SINR_DL_rzf_precoding(self, H, Hhat, realizations_range):
        #Store identity matrices of different sizes
        eyeK = np.eye(self.num_UEs_per_BS)
        #eyeM = np.eye(self.M)

        #Compute sum of all estimation error correlation matrices at every BS
        #C_totM = np.reshape(self.p*np.sum(np.sum(C,2),2),[self.M, self.M, self.L],'F')

        #Compute the prelog factor assuming only downlink transmission
        #prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)

        #Prepare to store simulation results for signal gains
        signal_RZF = np.zeros((self.num_UEs_per_BS, self.L), dtype=np.complex64)

        #Prepare to store simulation results for Bernoulli_extra_interference powers
        interf_RZF = np.zeros((self.num_UEs_per_BS, self.L, self.num_UEs_per_BS, self.L)) #real-valued, not complex

        #AK-TODO: make a smarter implementation
        number_of_realizations = 0
        for n in realizations_range: #range(self.nbrOfRealizations): #Go through all channel realizations
            number_of_realizations += 1

        for n in realizations_range: #range(self.nbrOfRealizations): #Go through all channel realizations
            #Go through all cells
            if n < 0 or n > self.nbrOfRealizations:
                raise Exception(str(n) + ' is outside the range [0, ' + str(self.nbrOfRealizations) + ']')
            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j
                Hallj = np.reshape(H[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Extract channel realizations from all UEs to BS j
                #Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')
                #note that we use n=0 as the estimate in the beginning of the coherence block, which will be used throughout the block
                Hhatallj = np.reshape(Hhat[:,0,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Compute MR combining in (4.11)
                V_MR = Hhatallj[:, self.num_UEs_per_BS * j:self.num_UEs_per_BS * (j + 1)]
                #print('AK', V_MR)

                #Compute M-MMSE combining in (4.7)
                #Backslash or matrix left division. If A is a square matrix, A\B is roughly the same as inv(A)*B, except it is computed in a different way
                #tempM = self.p*(self.p*(np.matmul(Hhatallj,np.conj(np.transpose(Hhatallj)))) + C_totM[:,:,j]+eyeM)
                #V_MMMSE = np.matmul(np.linalg.inv(tempM), V_MR)
                #Go through all UEs in cell j

                #Compute RZF combining in (4.9)
                #V_RZF = p*V_MR/(p*(V_MR'*V_MR)+eyeK);
                temp = self.p*np.matmul(np.matrix(V_MR).getH(), V_MR) + eyeK
                temp = np.linalg.inv(temp)
                V_RZF = np.matmul(self.p * V_MR, temp)

                for k in range(self.num_UEs_per_BS):
                    if np.linalg.norm(V_MR[:,k])>0:
                        #RZF precoding
                        w = V_RZF[:,k]/np.linalg.norm(V_RZF[:,k]) #Extract precoding vector
                        w = np.reshape(w, (1,self.num_bs_antennas)).conj() #Hermitian: make it a row vector and conjugate

                        #Compute realizations of the terms inside the expectations
                        #of the signal and Bernoulli_extra_interference terms of (7.2) and (7.3)
                        h_temp = H[:,n,k,j,j]
                        signal_RZF[k,j] = signal_RZF[k,j] + (np.inner(w,h_temp))/number_of_realizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp, (self.num_UEs_per_BS, self.L), 'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_RZF[k,j,:,:] = interf_RZF[k,j,:,:] + h_temp/number_of_realizations
        #Compute the terms in (7.2)
        signal_RZF = np.abs(signal_RZF)**2
        #print('AK2', signal_RZF)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.num_UEs_per_BS):
                interf_RZF[k,j,k,j] = interf_RZF[k,j,k,j] - signal_RZF[k,j]
        return signal_RZF[:,self.target_BS], interf_RZF

    '''
    %INPUT:
    %rho          = K x L matrix where element (k,j) is the downlink transmit
    %               power allocated to UE k in cell j
    %signal       = K x L matrix where element (k,j) is a_jk in (7.2)
    %Bernoulli_extra_interference = K x L x K x L matrix where (l,i,j,k) is b_lijk in (7.3)
    %prelogFactor = Prelog factor
    %
    %OUTPUT:
    %SE = K x L matrix where element (k,j) is the downlink SE of UE k in cell j
    %     using the power allocation given as input
    From functionComputeSE_DL_poweralloc.m
    '''
    def computeSE_DL_poweralloc(self, rho, signal, interference):
        #AK-TODO rho is a matrix but could be a vector given that we are looking at the target BS only
        #Compute the prelog factor assuming only downlink transmission
        prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)
        #Prepare to save results
        SE = np.zeros((self.num_UEs_per_BS,))
        # Go through all cells
        #for j in range(self.L):
        #Go through all UEs in cell j
        j=self.target_BS
        for k in range(self.num_UEs_per_BS):
            #Compute the SE in Theorem 4.6 using the formulation in (7.1)
            SE[k] = prelogFactor*np.log2(1+(rho[k,j]*signal[k]) / (sum(sum(rho*interference[:,:,k,j])) + 1))
        return SE

    def get_number_of_states(self):
        return 10

    def get_number_of_actions(self):
        return 10

    def generate_SE_statistics(self):
        '''
        We may need to design quantizers, so this provides dynamic ranges.
        AK-TODO need to finish to make automatic, I will simply use to get numbers manually (1 to 9 bits for a
        1-bit quantizer). The max gets larger when we select users, but then the min is 0
        '''
        #test_estimation()
        num_realizations_per_episode = 300
        self.set_num_realizations(num_realizations_per_episode)
        H = self.channel_realizations()

        SE_all = np.zeros((num_realizations_per_episode, self.num_connected_users_in_target_BS))

        if self.channel_estimation_method == 'MMSE':
            if self.precoding_method == 'MMSE':
                Hhat_MMSE, C_MMSE = self.channel_estimates_mmse(H)
            else:
                Hhat_MMSE = self.channel_estimates_mmse_version_without_covariance(H)
        else:
            raise Exception('Method not implemented!')

        for n in range(num_realizations_per_episode):
            #two options:
            get_minimum_value = False
            if get_minimum_value:
                #selected_users = np.arange(self.num_connected_users_in_target_BS)
                H_selected = H
                Hhat_selected = Hhat_MMSE
            else:
                selected_users = np.random.choice(self.num_connected_users_in_target_BS,
                                                  self.Kmax, replace=False) #Matlab's randsample
                #eliminating influence by zero'ing the channels of non-selected users
                #print(selected_users) #for example, generate 0,1 or 2,1 when choosing from 3
                H_selected, Hhat_selected = zeroing_nonused_channels(H, Hhat_MMSE, selected_users)

            SE_evaluation_window = 1 #use 1 if no look into past
            realizations_range = np.arange(n,n+SE_evaluation_window)

            #precoding and combining methods: MMSE, LS or RZF
            if self.precoding_method == 'MMSE':
                signal_MMMSE, interf_MMMSE = self.compute_SINR_DL_mmse_precoding(H_selected, Hhat_selected, C_MMSE, realizations_range)
                SE = self.computeSE_DL_poweralloc(self.rhoEqual, signal_MMMSE, interf_MMMSE)
            elif self.precoding_method == 'MR':
                signal_MR, interf_MR = self.compute_SINR_DL_mr_precoding(H_selected, Hhat_selected, realizations_range)
                SE = self.computeSE_DL_poweralloc(self.rhoEqual, signal_MR, interf_MR)
            elif self.precoding_method == 'RZF':
                signal_RZF, interf_RZF = self.compute_SINR_DL_rzf_precoding(H_selected, Hhat_selected, realizations_range)
                SE = self.computeSE_DL_poweralloc(self.rhoEqual, signal_RZF, interf_RZF)
            else:
                raise Exception('Method not implemented!')

            #print('SE=', SE)
            SE_all[n] = SE
        return SE_all

    def convert_H_into_Hhat(self,H,debugme):
        #Generate realizations of normalized noise
        #the last dimension of Np in self.pilot_reuse_factor, not self.num_UEs_per_BS
        if debugme == True: #match other functions, for debugging
            Np = np.sqrt(0.5)*(np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor) +
                               1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor))
            Np = Np[:,0,:,:,:]
            Np = np.expand_dims(Np, axis=1)
        else:
            #this may be different from the ones in self.
            [num_bs_antennas,nbrOfRealizations,num_UEs_per_BS,L,Lrepeated]=H.shape
            #use only self.pilot_reuse_factor from self
            Np = np.sqrt(0.5)*(np.random.randn(num_bs_antennas, nbrOfRealizations, num_UEs_per_BS, L, self.pilot_reuse_factor) +
                               1j * np.random.randn(num_bs_antennas, nbrOfRealizations, num_UEs_per_BS, L, self.pilot_reuse_factor))

        #Prepare to store MMSE channel estimates
        Hhat_MMSE = np.zeros(H.shape, dtype=np.complex64)
        # Go through all cells
        for j in range(self.L):
            #Go through all f pilot groups
            for g in range(self.pilot_reuse_factor):
                #Extract the cells that belong to pilot group g
                groupMembers = np.where(self.pilotPattern == g+1) #add 1 because first pilot group is 1
                groupMembers = groupMembers[1] #AK-TODO buggy, fix it, now need to discard 1st dimension
                #if not groupMembers.any():
                #    raise RuntimeError('No cell is assigned to pilot group = ', g+1)
                #Compute processed pilot signal for all UEs that use these pilots, according to (3.5)

                Htemp = H[:,:,:,groupMembers,j]
                yp = np.sqrt(self.p)*self.tau_p*np.sum(Htemp,3) + np.sqrt(self.tau_p)*Np[:,:,:,j,g]
                for k in range(self.num_UEs_per_BS):
                    for l in groupMembers:
                        Hhat_MMSE[:,:,k,l,j] = np.matmul(self.RPsi[k,l,j],yp[:,:,k])
        return Hhat_MMSE

    '''
    Assume the following:
      For each coherence block:
        * generate num_tti + 1 channels H
        * convert the first channel into Hhat, which will be used in the whole block
            For each TTI:
                # use H[2:] and Hhat to estimate SE, etc.
    '''
    def prepare_H_Hhat_for_coherence_block(self, H, debugme):
        #first_H_channel = np.zeros((self.num_bs_antennas, 1, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)
        first_H_channel = H[:,0,:,:,:]
        #https://stackoverflow.com/questions/42312319/how-do-i-add-a-dimension-to-an-array-opposite-of-squeeze
        #first_H_channel = np.reshape(first_H_channel,(self.num_bs_antennas, 1, self.num_UEs_per_BS, self.L, self.L),'F')
        first_H_channel = np.reshape(first_H_channel,(self.num_bs_antennas, 1, self.num_UEs_per_BS, self.L, self.L))
        first_Hhat_channel = self.convert_H_into_Hhat(first_H_channel,debugme)
        self.Hhat = np.zeros(H.shape,dtype=np.complex64)
        #copy tensor. AK-TODO: improve speed, rewriting functions that use this
        for i in range(H.shape[1]):
            self.Hhat[:,i,:,:,:] = np.squeeze(first_Hhat_channel,1)
        self.H = H[:,1:,:,:,:]         #eliminate first channel from H and keep 2:end
        #return H[:,1:,:,:,:], first_Hhat_channel

    def SE_for_given_range_of_channels(self, n, selected_users, SE_evaluation_window = 1):
        #eliminating influence by zero'ing the channels of non-selected users
        #print(selected_users) #for example, generate 0,1 or 2,1 when choosing from 3

        #AK-TODO improve speed
        H_selected, Hhat_selected = zeroing_nonused_channels(self.H, self.Hhat, selected_users)

        #SE_evaluation_window = 1 #use 1 if no look into past
        realizations_range = np.arange(n,n+SE_evaluation_window)

        #precoding and combining methods: MMSE, LS or RZF
        if self.precoding_method == 'MMSE':
            signal_MMMSE, interf_MMMSE = self.compute_SINR_DL_mmse_precoding(H_selected, Hhat_selected, self.C_MMSE, realizations_range)
            SE = self.computeSE_DL_poweralloc(self.rhoEqual, signal_MMMSE, interf_MMMSE)
        elif self.precoding_method == 'MR':
            signal_MR, interf_MR = self.compute_SINR_DL_mr_precoding(H_selected, Hhat_selected, realizations_range)
            SE = self.computeSE_DL_poweralloc(self.rhoEqual, signal_MR, interf_MR)
        elif self.precoding_method == 'RZF':
            signal_RZF, interf_RZF = self.compute_SINR_DL_rzf_precoding(H_selected, Hhat_selected, realizations_range)
            SE = self.computeSE_DL_poweralloc(self.rhoEqual, signal_RZF, interf_RZF)
        else:
            raise Exception('Method not implemented!')

        return SE

def zeroing_nonused_channels(H, Hhat, selected_users):
    '''
    Assign zero to channels of non-selected users
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L.
    The vector H(:,n,k,j,l) is the n:th channel estimate of the channel between UE k in cell j
    and the BS in cell l.
    selected_users start with index 0
    '''
    H_selected = copy.deepcopy(H)
    Hhat_selected = copy.deepcopy(Hhat)

    N=H.shape[2]
    non_selected = np.ones((N,),dtype=bool)
    non_selected[selected_users]=False

    H_selected[:,:,non_selected,:,:] = 0
    Hhat_selected[:,:,non_selected,:,:] = 0

    return H_selected, Hhat_selected

def show_SE_statistics():
    #test_estimation()
    frequency_index = 1
    massiveMIMOSystem = MassiveMIMOSystem(frequency_index)
    SE = massiveMIMOSystem.generate_SE_statistics()
    print('SE min=', np.min(SE), 'max=', np.max(SE), 'std=', np.std(SE))
    sys.stdout = open('se_values.txt', 'w')
    SE = SE.flatten()
    for s in SE:
        print(s)

'''
Here I don't get SE for one specific set of channels (realization)
'''
def test_RL_with_batch_of_channels():
    #test_estimation()
    frequency_index = 3
    massiveMIMOSystem = MassiveMIMOSystem(frequency_index)
    num_ttis_per_coherence_block = 100
    massiveMIMOSystem.set_num_realizations(num_ttis_per_coherence_block)
    H = massiveMIMOSystem.channel_realizations()

    SE_averages = np.zeros((massiveMIMOSystem.num_connected_users_in_target_BS,))

    if massiveMIMOSystem.channel_estimation_method == 'MMSE':
        if massiveMIMOSystem.precoding_method == 'MMSE':
            Hhat_MMSE, C_MMSE = massiveMIMOSystem.channel_estimates_mmse(H)
        else:
            Hhat_MMSE = massiveMIMOSystem.channel_estimates_mmse_version_without_covariance(H)
    else:
        raise Exception('Method not implemented!')

    for n in range(num_ttis_per_coherence_block):
        #this is a decision from RL agent
        selected_users = np.random.choice(massiveMIMOSystem.num_connected_users_in_target_BS,
                                          massiveMIMOSystem.Kmax, replace=False) #Matlab's randsample
        #eliminating influence by zero'ing the channels of non-selected users
        print(selected_users) #for example, generate 0,1 or 2,1 when choosing from 3

        H_selected, Hhat_selected = zeroing_nonused_channels(H, Hhat_MMSE, selected_users)

        SE_evaluation_window = 1 #use 1 if no look into past
        realizations_range = np.arange(n,n+SE_evaluation_window)

        #precoding and combining methods: MMSE, LS or RZF
        if massiveMIMOSystem.precoding_method == 'MMSE':
            signal_MMMSE, interf_MMMSE = massiveMIMOSystem.compute_SINR_DL_mmse_precoding(H_selected, Hhat_selected, C_MMSE, realizations_range)
            SE = massiveMIMOSystem.computeSE_DL_poweralloc(massiveMIMOSystem.rhoEqual, signal_MMMSE, interf_MMMSE)
        elif massiveMIMOSystem.precoding_method == 'MR':
            signal_MR, interf_MR = massiveMIMOSystem.compute_SINR_DL_mr_precoding(H_selected, Hhat_selected, realizations_range)
            SE = massiveMIMOSystem.computeSE_DL_poweralloc(massiveMIMOSystem.rhoEqual, signal_MR, interf_MR)
        elif massiveMIMOSystem.precoding_method == 'RZF':
            signal_RZF, interf_RZF = massiveMIMOSystem.compute_SINR_DL_rzf_precoding(H_selected, Hhat_selected, realizations_range)
            SE = massiveMIMOSystem.computeSE_DL_poweralloc(massiveMIMOSystem.rhoEqual, signal_RZF, interf_RZF)
        else:
            raise Exception('Method not implemented!')

        print('SE=', SE)
        SE_averages += SE
    print('SE_averages=', SE_averages / num_ttis_per_coherence_block)

def compare_new_approach():
    #test_estimation()
    frequency_index = 3
    massiveMIMOSystem = MassiveMIMOSystem(frequency_index)
    num_ttis_per_coherence_block = 100
    massiveMIMOSystem.set_num_realizations(num_ttis_per_coherence_block)
    H = massiveMIMOSystem.channel_realizations()
    np.random.seed(1)
    if massiveMIMOSystem.channel_estimation_method == 'MMSE':
        if massiveMIMOSystem.precoding_method == 'MMSE':
            Hhat_MMSE, C_MMSE = massiveMIMOSystem.channel_estimates_mmse(H)
        else:
            Hhat_MMSE = massiveMIMOSystem.channel_estimates_mmse_version_without_covariance(H)
    else:
        raise Exception('Method not implemented!')
    #make sure the random noise is the same by using the same seed
    np.random.seed(4)
    #Hhat_MMSE2, C_MMSE2 = massiveMIMOSystem.initialize_mmse_channel_estimator(H)

    if False:
        np.random.seed(4)
        Hhat_MMSE3 = massiveMIMOSystem.convert_H_into_Hhat(H)

        print(compare_tensors(Hhat_MMSE2, Hhat_MMSE3))
        print(compare_tensors(C_MMSE, C_MMSE2))

        debugme = True
        np.random.seed(4)
        massiveMIMOSystem.prepare_H_Hhat_for_coherence_block(H, debugme)
        debugme = False
        np.random.seed(4)
        massiveMIMOSystem.prepare_H_Hhat_for_coherence_block(H, debugme)

'''
Here I closely mimic what we need on RL and get SE for one specific set of channels (realization)
For each episode:
- I will assume the positions are fixed now (later I can generate new positions, i.e., use new .mat files)
- choose reasonable num_realizations_for_channel_estimation (number of observations) and calculate the
  matrices to generate channel estimates
  For each coherence block:
  * generate num_tti + 1 channels H
  * convert the first channel into Hhat, which will be used in the whole block
    For each TTI:
    # use H[2:] and Hhat to estimate SE, etc.

Here we use a single frequency band
'''
def test_RL():
    num_episodes = 3
    frequency_index = 1
    num_realizations_for_channel_estimation = 20
    num_ttis_per_coherence_block = 3
    num_coherence_blocks_per_episode = 4

    massiveMIMOSystem = MassiveMIMOSystem(frequency_index)

    for e in range(num_episodes):
        #initialize
        SE_averages = np.zeros((massiveMIMOSystem.num_connected_users_in_target_BS,))
        #calculate the matrices to generate channel estimates
        massiveMIMOSystem.set_num_realizations(num_realizations_for_channel_estimation)
        H = massiveMIMOSystem.channel_realizations()
        massiveMIMOSystem.initialize_mmse_channel_estimator(H)
        for b in range(num_coherence_blocks_per_episode):
            massiveMIMOSystem.set_num_realizations(num_ttis_per_coherence_block + 1)
            H = massiveMIMOSystem.channel_realizations()
            debugme=False
            massiveMIMOSystem.prepare_H_Hhat_for_coherence_block(H,debugme)
            for n in range(num_ttis_per_coherence_block):
                #this is a decision from RL agent
                selected_users = np.random.choice(massiveMIMOSystem.num_connected_users_in_target_BS,
                                                  massiveMIMOSystem.Kmax, replace=False) #Matlab's randsample
                SE = massiveMIMOSystem.SE_for_given_range_of_channels(n, selected_users, SE_evaluation_window = 1)
                #print('SE=', SE)
                SE_averages += SE
        print('SE_averages=', SE_averages / (num_coherence_blocks_per_episode*num_ttis_per_coherence_block))

if __name__ == '__main__':
    #https://stackoverflow.com/questions/38461588/achieve-same-random-numbers-in-numpy-as-matlab
    np.random.seed(1)
    test_RL()
    #show_SE_statistics()
    #compare_new_approach()