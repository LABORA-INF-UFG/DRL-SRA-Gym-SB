'''

Matches the info generated with ak_main_hope3.m.

Massive MIMO system.

AK-TODO: should crop the channel matrices instead of using zeros

- Compared to version 3: this one relies on channels pre-computed in Matlab.
- Not calling TTI but sample

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

from akpy.matlab_tofrom_python import read_several_matlab_arrays_from_mat, read_matlab_array_from_mat
#from .matlab_tofrom_python import read_matlab_array_from_mat
import numpy as np
from scipy.linalg import sqrtm
#import matplotlib.pyplot as plt
import os
import copy
import sys
from akpy.util import compare_tensors
from pathlib import Path

class MassiveMIMOSystem:
    def __init__(self, K=3, frequency_index=1):
        #depends on dataset
        #self.num_episodes = 200 ## 200 for new data
        self.num_episodes = 30
        self.range_episodes_train = [0,99]
        self.range_episodes_validate = [100,199]
        self.num_blocks_per_episode = 100 ## 100 for new data
        #self.num_blocks_per_episode = 20
        self.current_sample_index = 0
        #Select length of coherence block
        #Length of pilot sequences
        self.tau_p = 30 #samples dedicated to pilots in coherence block, self.pilot_reuse_factor * self.num_UEs_per_BS
        self.tau_u = 40
        self.tau_d = 140
        self.tau_c = self.tau_u + self.tau_p + self.tau_d #samples in coherence block (samples per frame)
        #channel estimation method: MMSE or
        self.channel_estimation_method = 'MMSE'
        self.H = None
        self.Hhat = None
        #precoding and combining methods: MMSE, MR or RZF
        self.precoding_method = 'MR'
        if os.name == 'nt':
            #self.root_folder = Path.cwd() / 'exp1'
            #self.root_folder = Path('E:\Docs_Doutorado') / 'exp3'
            #self.root_folder = Path('E:\Docs_Doutorado') / 'exp1_origin'
            self.root_folderb = Path('E:\Docs_Doutorado') / 'exp1_origin'
            self.root_folder = Path('E:\Docs_Doutorado') / 'exp1_f2_mmw'
        else:
            #self.root_folder = '/mnt/c/aksimuls/exp1/'  #laptop
            self.root_folder = Path.cwd() / 'exp1'   #UT PC
        self.file_name_prefix = self.root_folder / 'channels' / ('channels_f' + str(frequency_index) + '_b' + str(self.num_blocks_per_episode) + 'tauc' + str(self.tau_c) + 'taup' + str(self.tau_p) + 'e_')

        #number of connected UEs. Cannot be larger than num_UEs_per_BS
        #to simplify, all cells have same number of UEs. But if
        #num_connected_users < num_UEs_per_BS, only the first elements
        #are valid for target_BS.
        #Number of UEs per BS
        self.num_UEs_per_BS = K #old K
        #below needs to be equals for now due to matrix dimensions
        self.num_connected_users_in_target_BS = self.num_UEs_per_BS
        #maximum number of served users
        #self.Kmax = Kmax
        #Number of BSs
        self.L = 19
        #index of target BS. Only one is studied
        self.target_BS = 1-1 #starts from index 0 and 5 is the center when L=9 and 1 when L=19
        #Number of BS antennas
        self.num_bs_antennas = 64 #old M
        #Define the pilot reuse factor
        self.pilot_reuse_factor = 3 #old f
        #Select the number of setups with random UE locations
        #nbrOfSetups = 100
        #Select the number of channel realizations per setup
        #self.nbrOfRealizations = 10
        ## Propagation parameters
        #Communication bandwidth
        self.BW = 100e6 #in Hz
        #self.BW = 20e6  # in Hz
        #from functionChannelEstimates.m

        #Total uplink transmit power per UE (mW)
        self.p = 100

        #Total downlink transmit power per UE (mW)
        rho = 100

        #load the gamma parameters
        gamma_file_name = self.root_folder / ('gamma_parameters'+str(frequency_index) + '.mat')#e.g. gamma_parameters1.mat
        self.gamma_parameters = read_matlab_array_from_mat(gamma_file_name, 'gamma_parameters')
        self.current_intercell_interference = np.zeros((self.num_UEs_per_BS,))
        #if we serve less users than K, interference should decrease:
        self.intercell_interference_scaling_factor = 1 #self.Kmax / self.num_UEs_per_BS

        #Maximum downlink transmit power per BS (mW)
        self.Pmax = self.num_UEs_per_BS * rho

        #Compute downlink power per UE in case of equal power allocation
        self.rhoEqual = (self.Pmax / self.num_UEs_per_BS)#AK: turned into a scalar, * np.ones((self.num_UEs_per_BS, self.L))

        #Define noise figure at BS (in dB)
        #noiseFigure = 7

        #Compute noise power
        #noiseVariancedBm = -174 + 10*np.log10(B) + noiseFigure

        #I am using ak_functionExampleSetup_resource_allocation.m
        #to generate R and channelGaindB
        #Use the approximation of the Gaussian local scattering model
        #accuracy = 2
        #Angular standard deviation in the local scattering model (in degrees)
        #ASDdeg = 10

        #Generate pilot pattern
        if self.pilot_reuse_factor == 1:
            self.pilotPattern = np.ones((self.L,))
        elif self.pilot_reuse_factor == 2: #Only works in the running example with its 16 BSs
            #self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1]).flatten()
            if self.L == 16:
                self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1])
            elif self.L == 9:
                self.pilotPattern = np.array([[1,2,1,2,1,2,1,2,1]])
        elif self.pilot_reuse_factor == 3: #Only works for example with 19 BSs
            self.pilotPattern = np.array([[1, 2, 3, 2, 3, 2, 3, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2]])
        elif self.pilot_reuse_factor == 4: #Only works in the running example with its 16 BSs
            #self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4]).flatten()
            self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4])
        elif self.pilot_reuse_factor == self.L:
            self.pilotPattern = np.arange(1,self.L)
        else:
            raise Exception('pilot_reuse_factor=', self.pilot_reuse_factor, 'not supported!')

        #print('Pilot groups=', self.pilotPattern)
        #Store identity matrix of size M x M
        self.eyeM = np.eye(self.num_bs_antennas)

    def get_current_income_packets(self):
        return self.num_pckts[:,self.current_sample_index]

    def get_current_Bernoulli_extra_interference(self):
        return self.Bernoulli_extra_interference[:, self.current_sample_index]

    def get_current_intercell_interference(self):
        return self.current_intercell_interference

    def update_intercell_interference(self):
        for u in range(self.num_UEs_per_BS):
            shape = self.gamma_parameters[u,0]
            scale = self.gamma_parameters[u,1]
            #s = np.random.gamma(shape, scale, 10000)
            self.current_intercell_interference[u] = self.intercell_interference_scaling_factor * np.random.gamma(shape, scale, 1)

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
    def computeSE_DL_poweralloc(self, rho, signal, interference, prelogFactor = 0.8421):
        #AK-TODO rho is a matrix but could be a vector given that we are looking at the target BS only
        #Compute the prelog factor assuming only downlink transmission
        #prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)
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

    #0.8421 considers tau_c = 190 and tau_p = 30
    def agora_vai_computeSE_DL_poweralloc(self, rho, signal, snr_denominator):
        prelogFactor = self.tau_d / self.tau_c
        #snr_denominator already incorporates rho
        #AK-TODO rho is a matrix but could be a vector given that we are looking at the target BS only
        #Compute the prelog factor assuming only downlink transmission
        #prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)
        #Prepare to save results
        num_users = len(signal)
        SE = np.zeros((num_users,))
        # Go through all cells
        #for j in range(self.L):
        #Go through all UEs in cell j
        j=self.target_BS
        for k in range(num_users):
            #Compute the SE in Theorem 4.6 using the formulation in (7.1)
            SE[k] = prelogFactor*np.log2(1 + (rho*signal[k] / snr_denominator[k]))
        return SE

    #def get_number_of_states(self):
    #    return 10

    #def get_number_of_actions(self):
    #    return 10

    def old_generate_SE_statistics(self):
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

    def SE_agora_vai(self, selected_users, H, Hhat, Kmax):
        #eliminating influence by zero'ing the channels of non-selected users
        #print(selected_users) #for example, generate 0,1 or 2,1 when choosing from 3
        #AK-TODO improve speed
        H, Hhat = zeroing_nonused_channels(H, Hhat, selected_users)
        #Prepare to store simulation results for signal gains
        #signal_MR = np.zeros((self.num_UEs_per_BS, self.L), dtype=np.complex64)
        signal_MR = np.zeros((len(selected_users),), dtype=np.complex64)
        intracell_interference = np.zeros((len(selected_users),)) #real number

        #Prepare to store simulation results for Bernoulli_extra_interference powers
        #interf_MMMSE = np.zeros((self.K,self.L,self.K,self.L),dtype=np.complex64)
        this_intecell_interference = self.current_intercell_interference
        this_intecell_interference = this_intecell_interference[selected_users] #keep only for selected_users

        # Extract channel realizations from all UEs to BS j, which includes all UEs in cells different than j
        #Hallj = np.reshape(H, (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

        # Extract channel realizations from all UEs to BS j
        # Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')
        # note that we use n=0 as the estimate in the beginning of the coherence block, which will be used throughout the block
        #Hhatallj = np.reshape(Hhat, (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

        # Compute MR combining in (4.11)
        # simply extract V_MR corresponding to cell j from Hhat
        #V_MR = Hhat[:, :, self.target_BS]
        V_MR = Hhat[:, :]
        # print('AK', V_MR)

        # Go through all UEs in target cell
        for i in range(len(selected_users)): #range(self.num_UEs_per_BS):
            k = selected_users[i]
            if np.linalg.norm(V_MR[:, k]) > 0:
                # MR precoding
                w = V_MR[:, k] / np.linalg.norm(V_MR[:, k])  # Extract precoding vector
                w = np.reshape(w, (1, self.num_bs_antennas)).conj()  # Hermitian: make it a row vector and conjugate

                # Compute realizations of the terms inside the expectations
                # of the signal and Bernoulli_extra_interference terms of (7.2) and (7.3)
                h_temp = H[:, k]
                signal_MR[i] = signal_MR[i] + (np.inner(w, h_temp))

                #add to pre-computed denominator the contribution of intracell Bernoulli_extra_interference
                for j in range(len(selected_users)): #range(self.num_UEs_per_BS):
                    if j == i:
                        continue
                    k2 = selected_users[j]
                    intracell_interference[i] += self.rhoEqual*np.abs(np.inner(w, H[:, k2]))**2
                # interf_MR[k,j,:,:] = interf_MR[k,j,:,:] + h_temp/number_of_realizations

        # Compute the terms in (7.2)
        signal_MR = np.real(np.abs(signal_MR) ** 2)
        # return only the signal for the target BS but all interference already scaled by rho
        # the signal has not been scaled by rho yet.
        this_interference = intracell_interference + self.rhoEqual * this_intecell_interference
        #prelogFactor = self.tau_d/self.tau_c
        power_per_selected_user = (Kmax/len(selected_users)) * self.rhoEqual #distribute among served
        SE = self.agora_vai_computeSE_DL_poweralloc(power_per_selected_user, signal_MR, this_interference)

        return SE

    def SE_for_given_range_of_channels(self, n, selected_users, H, Hhat, SE_evaluation_window = 1):
        #eliminating influence by zero'ing the channels of non-selected users
        #print(selected_users) #for example, generate 0,1 or 2,1 when choosing from 3

        #AK-TODO improve speed
        H_selected, Hhat_selected = zeroing_nonused_channels(H, Hhat, selected_users)

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

    '''
    File names start with 1 and episode_index with 0, so add 1
    '''
    def get_episode_channels(self):
        return self.H, self.Hhat

    def get_episode_interference_per_user(self):
        return None #self.gamma_parameters

    def load_episode(self, episode_number):
        if episode_number > self.num_episodes-1:
            raise Exception("episode_number > self.num_episodes-1:" + str(episode_number))
        self.episode_number = episode_number
        file_name = str(self.file_name_prefix) + (str(episode_number+1) + '.mat')
        arrayNames = ('H','Hhat')
        my_list = read_several_matlab_arrays_from_mat(file_name, arrayNames)
        self.H = my_list[0]
        self.Hhat = my_list[1]
        #self.snr_denominator = my_list[2]

        #AK-TODO we are wasting space here, given that the traffic files are the same. They should be
        #associated to another class, not here

        #assume a folder structure such that:
        file_name = self.root_folder / ('traffic_interference/traffic_interference_e_' + str(episode_number+1) + '.mat')
        arrayNames = ('num_pckts','interference')
        my_list = read_several_matlab_arrays_from_mat(file_name, arrayNames)
        self.num_pckts = my_list[0]
        self.Bernoulli_extra_interference = my_list[1]

        #make sure we reset index
        self.current_sample_index = 0

    def SE_for_given_sample(self, current_sample_index, selected_users, Kmax, avoid_estimation_errors = False):
        # initialize
        #SE_averages = np.zeros((self.num_connected_users_in_target_BS,))
        # calculate the matrices to generate channel estimates
        #print('current_sample_index, selected_users =', current_sample_index, selected_users)

        #b = int(current_sample_index / (self.tau_c + 1))
        b = int(current_sample_index / self.tau_d)

        thisHhat = self.Hhat[:, b, :]
        thisH = self.H[:, current_sample_index, :]

        if avoid_estimation_errors == True:
            #will avoid estimation errors
            thisHhat = thisH

        # this is a decision from RL agent
        #selected_users = np.random.choice(self.num_connected_users_in_target_BS,
        #                                  self.Kmax, replace=False)  # Matlab's randsample
        # SE = self.SE_for_given_range_of_channels(n, selected_users, thisH, thisHhat, SE_evaluation_window = 1)

        SE = self.SE_agora_vai(selected_users, thisH, thisHhat, Kmax)
        # print('SE=', SE)
        thisSE = np.zeros((self.num_connected_users_in_target_BS,))
        thisSE[selected_users] = SE
        return thisSE
        #SE_averages += thisSE

        #print('SE_averages=', SE_averages / (num_coherence_blocks_per_episode * num_samples_per_coherence_block))



def zeroing_nonused_channels(H, Hhat, selected_users):
    '''
    Assign zero to channels of non-selected users
    H is complex-valued 3-d tensor with dimension M x K x L.
    The vector H(:,k,j) is the n:th channel estimate of the channel between UE k in cell j
    and the target BS in cell l=0. Hhat is similar
    selected_users start with index 0
    '''
    H_selected = copy.deepcopy(H)
    Hhat_selected = copy.deepcopy(Hhat)

    N=H.shape[1]
    non_selected = np.ones((N,),dtype=bool)
    non_selected[selected_users]=False

    H_selected[:,non_selected] = 0
    Hhat_selected[:,non_selected] = 0

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
    num_episodes = 30
    frequency_index = 1
    K = 10
    Kmax = 2 #note that it's not defined in this classe
    #num_bs_antennas = 64

    massiveMIMOSystem = MassiveMIMOSystem(K=K, frequency_index=frequency_index)

    for e in range(num_episodes):
        #initialize
        SE_averages = np.zeros((massiveMIMOSystem.num_connected_users_in_target_BS,))
        #calculate the matrices to generate channel estimates
        massiveMIMOSystem.load_episode(e)
        H, Hhat = massiveMIMOSystem.get_episode_channels()
        num_coherence_blocks_per_episode = Hhat.shape[1]
        #num_samples_per_coherence_block = int ( (H.shape[1]-1)/num_coherence_blocks_per_episode )
        num_samples_per_coherence_block = int ( H.shape[1] / num_coherence_blocks_per_episode )

        #H = massiveMIMOSystem.channel_realizations()
        for b in range(num_coherence_blocks_per_episode):
            thisHhat = Hhat[:,b,:]
            for n in range(num_samples_per_coherence_block):
                current_channel_index = b*num_samples_per_coherence_block + n #first sample of block
                massiveMIMOSystem.update_intercell_interference()
                #current_channel_index += 1 #first sample in block regards the pilot, so skip it
                thisH = H[:,current_channel_index,:]
                #this is a decision from RL agent
                selected_users = np.random.choice(massiveMIMOSystem.num_connected_users_in_target_BS,
                                                  Kmax, replace=False) #Matlab's randsample
                #SE = massiveMIMOSystem.SE_for_given_range_of_channels(n, selected_users, thisH, thisHhat, SE_evaluation_window = 1)

                SE = massiveMIMOSystem.SE_agora_vai(selected_users, thisH, thisHhat, Kmax)
                #print('SE=', SE)
                thisSE = np.zeros((massiveMIMOSystem.num_connected_users_in_target_BS,))
                thisSE[selected_users] = SE
                SE_averages += thisSE
        print('SE_averages=', SE_averages / (num_coherence_blocks_per_episode*num_samples_per_coherence_block))

def test_intercell_interference():
    massiveMIMOSystem = MassiveMIMOSystem(K=10, frequency_index=1)
    for i in range(100):
        massiveMIMOSystem.update_intercell_interference()
        print(massiveMIMOSystem.get_current_intercell_interference())

if __name__ == '__main__':
    #test_intercell_interference()
    #exit(1)
    #shape, scale = 2., 4.  # mean=4, std=2*sqrt(2)
    #s = np.random.gamma(shape, scale, 10000)
    #print(np.mean(s))
    #exit(1)
    #https://stackoverflow.com/questions/38461588/achieve-same-random-numbers-in-numpy-as-matlab
    np.random.seed(1)
    test_RL()
    #show_SE_statistics()
    #compare_new_approach()
