'''
Massive MIMO system.

It is missing the capability of dealing with 2 frequencies.

Based on:
%Emil Bjornson, Jakob Hoydis and Luca Sanguinetti (2017),
%"Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency",
%Foundations and Trends in Signal Processing: Vol. 11, No. 3-4,
%pp. 154-655. DOI: 10.1561/2000000093.
%https://www.massivemimobook.com
'''
#in all reshapes I need to use 'F': h_temp = np.reshape(h_temp,(self.K,self.L),'F')

from akpy.matlab_tofrom_python import read_matlab_array_from_mat
#from .matlab_tofrom_python import read_matlab_array_from_mat
import numpy as np
from scipy.linalg import sqrtm
#import matplotlib.pyplot as plt
import os

class MassiveMIMOSystem:
    def __init__(self):
        #channel estimation method: MMSE or
        self.channel_estimation_method = 'MMSE'
        #precoding and combining methods: MMSE, MR or RZF
        self.precoding_method = 'MR'
        #index of target BS. Only one is studied
        self.target_BS = 7-1 #starts from index 0
        #number of connected UEs. Cannot be larger than num_UEs_per_BS
        #to simplify, all cells have same number of UEs. But if
        #num_connected_users < num_UEs_per_BS, only the first elements
        #are valid for target_BS.
        self.num_connected_users_in_target_BS = 3
        #maximum number of served users
        self.Kmax = 2
        #Number of BSs
        self.L = 16
        #Number of UEs per BS
        self.num_UEs_per_BS = 3 #old K
        #Number of BS antennas
        self.num_bs_antennas = 64 #old M
        #Define the pilot reuse factor
        self.pilot_reuse_factor = 2 #old f
        #Select the number of setups with random UE locations
        #nbrOfSetups = 100
        #Select the number of channel realizations per setup
        self.nbrOfRealizations = 100
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

        #Use the approximation of the Gaussian local scattering model
        accuracy = 2

        #Angular standard deviation in the local scattering model (in degrees)
        ASDdeg = 10
        if os.name == 'nt':
            fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/R_channelGaindB.mat'
        else:
            fileName = '/mnt/d/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/R_channelGaindB.mat'
        self.channelGaindB=read_matlab_array_from_mat(fileName, 'channelGaindB')
        self.R=read_matlab_array_from_mat(fileName, 'R') #normalized to have norm = M

        #print('R', self.R.shape)

        #Compute the normalized average channel gain, where the normalization
        #is based on the noise power
        channelGainOverNoise = self.channelGaindB - noiseVariancedBm

        self.channel_gain_over_noise_linear = 10**(channelGainOverNoise/10)

        #from functionChannelEstimates.m
        #Length of pilot sequences
        self.tau_p = self.pilot_reuse_factor * self.num_UEs_per_BS

        #Generate pilot pattern
        if self.pilot_reuse_factor == 1:
            self.pilotPattern = np.ones((self.L,))
        elif self.pilot_reuse_factor == 2: #Only works in the running example with its 16 BSs
            #self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1]).flatten()
            self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 2, 1, 2, 1])
        elif self.pilot_reuse_factor == 4: #Only works in the running example with its 16 BSs
            #self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4]).flatten()
            self.pilotPattern = np.kron([[1,1]],[1, 2, 1, 2, 3, 4, 3, 4])
        elif self.pilot_reuse_factor == 16: #Only works in the running example with its 16 BSs
            self.pilotPattern = np.arange(1,self.L)

        print('Pilot groups=', self.pilotPattern)
        #Store identity matrix of size M x M
        self.eyeM = np.eye(self.num_bs_antennas)

    '''
    Get self.nbrOfRealizations realizations of all channels.
    H is complex-valued 5-d tensor with dimension M x self.nbrOfRealizations x K x L x L.
    '''
    def channel_realizations(self):
        #Go through all channels and apply the channel gains to the spatial
        H = np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L) + \
                1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L)

        for j in range(self.L):
            for l in range(self.L):
                for k in range(self.num_UEs_per_BS):
                    Rtemp = self.channel_gain_over_noise_linear[k,j,l] * self.R[:,:,k,j,l]
                    #print(Rtemp.shape)
                    #Rtemp=np.matrix([[8,1+3j,6],[3-100j,5,7],[4,9,2]])
                    Rsqrt= sqrtm(Rtemp)
                    #print(Rsqrt)
                    #exit(-1)
                    Htemp = H[:,:,k,j,l]
                    #Apply correlation to the uncorrelated channel realizations
                    H[:,:,k,j,l] = np.sqrt(0.5) * np.matmul(Rsqrt,Htemp)
        return H

    '''
    Linear MMSE channel estimator. From function ak_functionChannelEstimates.m
    Need to implement the EW-MMSE estimator and the LS estimator.
    '''
    def channel_estimates_mmse(self,H):
        #Generate realizations of normalized noise
        Np = np.sqrt(0.5)*(np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor) +
                                   1j * np.random.randn(self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.pilot_reuse_factor))
        #Prepare to store MMSE channel estimates
        Hhat_MMSE = np.zeros((self.num_bs_antennas, self.nbrOfRealizations, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)
        #Prepare to store estimation error correlation matrices
        C_MMSE = np.zeros((self.num_bs_antennas, self.num_bs_antennas, self.num_UEs_per_BS, self.L, self.L), dtype=np.complex64)

        #Go through all channels and apply the channel gains to the spatial
        #correlation matrices
        R_scaled = np.zeros(self.R.shape, dtype=np.complex64)
        for j in range(self.L):
            for l in range(self.L):
                for k in range(self.num_UEs_per_BS):
                    if self.channelGaindB[k,j,l]>-np.Inf:
                        #Extract channel gain in linear scale
                        #channel_gain_linear = 10**(self.channelGaindB[k,j,l]/10)
                        #Apply channel gain to correlation matrix
                        R_scaled[:,:,k,j,l] = self.R[:,:,k,j,l] * self.channel_gain_over_noise_linear[k,j,l]
                        #R_scaled[:,:,k,j,l] = self.R[:,:,k,j,l] * channel_gain_linear
                    else:
                        R_scaled[:,:,k,j,l] = 0
                        H[:,:,k,j,l] = 0

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
                    Rtemp = R_scaled[:,:,k,groupMembers,j]
                    PsiInv = (self.p*self.tau_p*np.sum(Rtemp,2) + self.eyeM)
                    for l in groupMembers:
                        #Compute MMSE estimate of channel between BS l and UE k in
                        #cell j using (3.9) in Theorem 3.1
                        #x = B/A is the solution to the equation xA = B. mrdivide in Matlab
                        RPsi = np.matmul(R_scaled[:,:,k,l,j], np.linalg.inv(PsiInv))
                        Hhat_MMSE[:,:,k,l,j] = np.sqrt(self.p)*np.matmul(RPsi,yp[:,:,k])
                        #Compute corresponding estimation error correlation matrix, using (3.11)
                        C_MMSE[:,:,k,l,j] = R_scaled[:,:,k,l,j] - self.p*self.tau_p*np.matmul(RPsi,R_scaled[:,:,k,l,j])
        return Hhat_MMSE, C_MMSE


    '''
    from functionComputeSINR_DL.m
    Implements only the MMSE precoding (from MMSE combining and duality)
    '''
    def compute_SINR_DL_mmse_precoding(self, H, Hhat, C):
        if False:
            #fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
            fileName = '/mnt/d/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
            H=read_matlab_array_from_mat(fileName, 'H')
            Hhat=read_matlab_array_from_mat(fileName, 'Hhat')
            C=read_matlab_array_from_mat(fileName, 'C')

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

        for n in range(self.nbrOfRealizations): #Go through all channel realizations
            #Go through all cells
            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j
                Hallj = np.reshape(H[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Extract channel realizations from all UEs to BS j
                Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

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
                        signal_MMMSE[k,j] = signal_MMMSE[k,j] + (np.inner(w,h_temp))/self.nbrOfRealizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp, (self.num_UEs_per_BS, self.L), 'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_MMMSE[k,j,:,:] = interf_MMMSE[k,j,:,:] + h_temp/self.nbrOfRealizations
        #Compute the terms in (7.2)
        signal_MMMSE = np.abs(signal_MMMSE)**2
        #print('AK2', signal_MMMSE)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.num_UEs_per_BS):
                interf_MMMSE[k,j,k,j] = interf_MMMSE[k,j,k,j] - signal_MMMSE[k,j]
        return signal_MMMSE, interf_MMMSE

    '''
    from functionComputeSINR_DL.m
    Implements only the MR precoding (from MR combining and duality)
    '''
    def compute_SINR_DL_mr_precoding(self, H, Hhat, C):
        if False:
            #fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
            fileName = '/mnt/d/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
            H=read_matlab_array_from_mat(fileName, 'H')
            Hhat=read_matlab_array_from_mat(fileName, 'Hhat')
            C=read_matlab_array_from_mat(fileName, 'C')

        #Store identity matrices of different sizes
        #eyeK = np.eye(self.K)
        #eyeM = np.eye(self.M)

        #Compute sum of all estimation error correlation matrices at every BS
        #C_totM = np.reshape(self.p*np.sum(np.sum(C,2),2),[self.M, self.M, self.L],'F')

        #Compute the prelog factor assuming only downlink transmission
        #prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)

        #Prepare to store simulation results for signal gains
        signal_MR = np.zeros((self.num_UEs_per_BS, self.L), dtype=np.complex64)

        #Prepare to store simulation results for Bernoulli_extra_interference powers
        #interf_MMMSE = np.zeros((self.K,self.L,self.K,self.L),dtype=np.complex64)
        interf_MR = np.zeros((self.num_UEs_per_BS, self.L, self.num_UEs_per_BS, self.L)) #real-valued, not complex

        for n in range(self.nbrOfRealizations): #Go through all channel realizations
            #Go through all cells
            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j
                Hallj = np.reshape(H[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Extract channel realizations from all UEs to BS j
                Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Compute MR combining in (4.11)
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
                        signal_MR[k,j] = signal_MR[k,j] + (np.inner(w,h_temp))/self.nbrOfRealizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp, (self.num_UEs_per_BS, self.L), 'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_MR[k,j,:,:] = interf_MR[k,j,:,:] + h_temp/self.nbrOfRealizations
        #Compute the terms in (7.2)
        signal_MR = np.abs(signal_MR)**2
        #print('AK2', signal_MR)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.num_UEs_per_BS):
                interf_MR[k,j,k,j] = interf_MR[k,j,k,j] - signal_MR[k,j]
        return signal_MR, interf_MR


    '''
    from functionComputeSINR_DL.m
    Implements only the RZF precoding (from RZF combining and duality)
    '''
    def compute_SINR_DL_rzf_precoding(self, H, Hhat, C):
        if False:
            #fileName = 'D:/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
            fileName = '/mnt/d/gits/lasse/software/mimo-toolbox/third_party/emil_massivemimobook/Code/H_Hhat_C.mat'
            H=read_matlab_array_from_mat(fileName, 'H')
            Hhat=read_matlab_array_from_mat(fileName, 'Hhat')
            C=read_matlab_array_from_mat(fileName, 'C')

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

        for n in range(self.nbrOfRealizations): #Go through all channel realizations
            #Go through all cells
            for j in range(self.L):
                #Extract channel realizations from all UEs to BS j
                Hallj = np.reshape(H[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

                #Extract channel realizations from all UEs to BS j
                Hhatallj = np.reshape(Hhat[:,n,:,:,j], (self.num_bs_antennas, self.num_UEs_per_BS * self.L), 'F')

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
                        signal_RZF[k,j] = signal_RZF[k,j] + (np.inner(w,h_temp))/self.nbrOfRealizations
                        h_temp = np.matmul(w,Hallj)
                        h_temp = np.abs(np.array(h_temp))**2
                        h_temp = np.reshape(h_temp, (self.num_UEs_per_BS, self.L), 'F')

                        #print('AK', np.max(h_temp[:]))

                        interf_RZF[k,j,:,:] = interf_RZF[k,j,:,:] + h_temp/self.nbrOfRealizations
        #Compute the terms in (7.2)
        signal_RZF = np.abs(signal_RZF)**2
        #print('AK2', signal_RZF)
        #Compute the terms in (7.3)
        for j in range(self.L):
            for k in range(self.num_UEs_per_BS):
                interf_RZF[k,j,k,j] = interf_RZF[k,j,k,j] - signal_RZF[k,j]
        return signal_RZF, interf_RZF


    '''
    %INPUT:
    %rho          = K x L matrix where element (k,j) is the downlink transmit
    %               power allocated to UE k in cell j
    %signal       = K x L matrix where element (k,j,n) is a_jk in (7.2)
    %Bernoulli_extra_interference = K x L x K x L matrix where (l,i,jk,n) is b_lijk in (7.3)
    %prelogFactor = Prelog factor
    %
    %OUTPUT:
    %SE = K x L matrix where element (k,j) is the downlink SE of UE k in cell j
    %     using the power allocation given as input
    '''
    def computeSE_DL_poweralloc(self, rho, signal, interference):
        #Compute the prelog factor assuming only downlink transmission
        prelogFactor = (self.tau_c-self.tau_p)/(self.tau_c)
        #Prepare to save results
        SE = np.zeros((self.num_UEs_per_BS, self.L))
        # Go through all cells
        for j in range(self.L):
            #Go through all UEs in cell j
            for k in range(self.num_UEs_per_BS):
                #Compute the SE in Theorem 4.6 using the formulation in (7.1)
                SE[k,j] = prelogFactor*np.log2(1+(rho[k,j]*signal[k,j]) / (sum(sum(rho*interference[:,:,k,j])) + 1))
        return SE

if __name__ == '__main__':
    #https://stackoverflow.com/questions/38461588/achieve-same-random-numbers-in-numpy-as-matlab
    np.random.seed(1)

    #test_estimation()
    massiveMIMOSystem = MassiveMIMOSystem()
    H = massiveMIMOSystem.channel_realizations()

    if massiveMIMOSystem.channel_estimation_method == 'MMSE':
        Hhat_MMSE, C_MMSE = massiveMIMOSystem.channel_estimates_mmse(H)
    else:
        raise Exception('Method not implemented!')

    #precoding and combining methods: MMSE, LS or RZF
    if massiveMIMOSystem.precoding_method == 'MMSE':
        signal_MMMSE, interf_MMMSE = massiveMIMOSystem.compute_SINR_DL_mmse_precoding(H, Hhat_MMSE, C_MMSE)
        SE = massiveMIMOSystem.computeSE_DL_poweralloc(massiveMIMOSystem.rhoEqual, signal_MMMSE, interf_MMMSE)
    elif massiveMIMOSystem.precoding_method == 'MR':
        signal_MR, interf_MR = massiveMIMOSystem.compute_SINR_DL_mr_precoding(H, Hhat_MMSE, C_MMSE)
        SE = massiveMIMOSystem.computeSE_DL_poweralloc(massiveMIMOSystem.rhoEqual, signal_MR, interf_MR)
        print(signal_MR)
        print(interf_MR[0])
    elif massiveMIMOSystem.precoding_method == 'RZF':
        signal_RZF, interf_RZF = massiveMIMOSystem.compute_SINR_DL_rzf_precoding(H, Hhat_MMSE, C_MMSE)
        SE = massiveMIMOSystem.computeSE_DL_poweralloc(massiveMIMOSystem.rhoEqual, signal_RZF, interf_RZF)
    else:
        raise Exception('Method not implemented!')

    print('SE=', SE)
