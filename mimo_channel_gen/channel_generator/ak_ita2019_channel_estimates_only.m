function [Hhat_MMSE,C_MMSE,Hhat_EW_MMSE,C_EW_MMSE,Hhat_LS,C_LS] = ak_ita2019_channel_estimates_only(R,H,nbrOfRealizations,p,f,tau_p)
%only generates estimation for given channels
%AK: this functions concerns only the channels to the target cell
%
%Generate the channel realizations and estimates of these channels for all
%UEs in the entire network. The channels are assumed to be correlated
%Rayleigh fading. The MMSE estimator, EW-MMSE estimator, and LS estimator
%are used. The latter two estimators are only computed if their outputs are
%requested when calling the function.
%
%INPUT:
%R                 = M x M x K x L  matrix with spatial correlation
%                    matrices for all UEs in the network. R(:,:,k,j) is
%                    the correlation matrix for the channel between UE k
%                    in cell j and the target BS (first cell). This matrix can
%                    either include the average channel gain or can be
%                    normalized arbitrarily.
%channelGaindB     = K x L matrix containing the average channel gains
%                    in dB of all the channels to target BS, if these are not already
%                    included in the spatial channel correlation matrices.
%                    The product R(:,:,k,j,l)*10^(channelGaindB(k,j,l)/10)
%                    is the full spatial channel correlation matrix.
%nbrOfRealizations = Number of channel realizations
%M                 = Number of antennas per BS
%K                 = Number of UEs per cell
%L                 = Number of BSs and cells
%p                 = Uplink transmit power per UE (same for everyone)
%f                 = Pilot reuse factor
%
%OUTPUT:
%Hhat_MMSE    = M x nbrOfRealizations x K x L matrix with the MMSE
%               channel estimates. The matrix Hhat_MMSE(:,n,k,j,l) is the
%               n:th channel estimate of the channel between UE k in cell j
%               and the target BS.
%C_MMSE       = M x M x K x L matrix with estimation error correlation
%               matrices when using MMSE estimation. The matrix is
%               organized in the same way as R.
%tau_p        = Length of pilot sequences
%R            = Scaled version of the input spatial correlation matrices R,
%               where the channel gains from channelGaindB are included
%H            = M x nbrOfRealizations x K x L  matrix with the true
%               channel realizations. The matrix is organized in the same
%               way as Hhat_MMSE.
%Hhat_EW_MMSE = Same as Hhat_MMSE, but using the EW-MMSE estimator
%C_EW_MMSE    = Same as C_MMSE, but using the EW-MMSE estimator
%Hhat_LS      = Same as Hhat_MMSE, but using the LS estimator
%C_LS         = Same as C_MMSE, but using the LS estimator
%
%
%This Matlab function was developed to generate simulation results to:
%
%Emil Bjornson, Jakob Hoydis and Luca Sanguinetti (2017),
%"Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency",
%Foundations and Trends in Signal Processing: Vol. 11, No. 3-4,
%pp. 154-655. DOI: 10.1561/2000000093.
%
%For further information, visit: https://www.massivemimobook.com
%
%This is version 1.0 (Last edited: 2017-11-04)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%monograph as described above.

L=size(R,4); %number of base station (BS)
K=size(R,3); %number of UE per BS
M=size(R,1); %number of antennas at BS (number at UE is always 1)

%Prepare to store MMSE channel estimates
Hhat_MMSE = zeros(M,nbrOfRealizations,K,L);

%Prepare to store estimation error correlation matrices
C_MMSE = zeros(M,M,K,L);
use_rand_for_python = 0;

%% Perform channel estimation

%Generate pilot pattern
if f == 1
    
    pilotPattern = ones(L,1);
    
elseif f == 2 %Only works in the running example with its 16 BSs
    
    pilotPattern = kron(ones(2,1),[1; 2; 1; 2; 2; 1; 2; 1]);
    
elseif f == 3 %Only works in the example with L=19 hexagonal BSs    
    %group 1: 1,8,10,12,14,16,18
    %group 2: 2,4,6,11,15,19
    %group 3: 3,5,7,9,13,17
    pilotPattern = [1 2 3 2 3 2 3 1 3 1 2 1 3 1 2 1 3 1 2]';
        
elseif f == 4 %Only works in the running example with its 16 BSs
    
    pilotPattern = kron(ones(2,1),[1; 2; 1; 2; 3; 4; 3; 4]);
    
elseif f == L %do not reuse pilots
    
    pilotPattern = (1:L)';
    
end


%Store identity matrix of size M x M
eyeM = eye(M);

%Generate realizations of normalized noise
if use_rand_for_python == 1
    Np = sqrt(0.5)*(rand(M,nbrOfRealizations,K,f) + 1i*rand(M,nbrOfRealizations,K,f));
else
    if 1
        Np = sqrt(0.5)*(randn(M,nbrOfRealizations,K,f) + 1i*randn(M,nbrOfRealizations,K,f));
    else %for debugging only
        Np = sqrt(0.5)*(randn(M,nbrOfRealizations,K,L,f) + 1i*randn(M,nbrOfRealizations,K,L,f));
        Np = squeeze(Np(:,:,:,1,:));
    end
end

%save -v6 'H_Np_R.mat' H Np R %debugging in Python

%Prepare for MMSE estimation

%Prepare for EW-MMSE estimation
if nargout >= 3
    
    %Prepare to store EW-MMSE channel estimates
    Hhat_EW_MMSE = zeros(M,nbrOfRealizations,K,L);
    
    %Prepare to store estimation error correlation matrices
    C_EW_MMSE = zeros(M,M,K,L);
    
end

%Prepare for LS estimation
if nargout >= 5
    
    %Prepare to store EW-MMSE channel estimates
    Hhat_LS = zeros(M,nbrOfRealizations,K,L);
    
    %Prepare to store estimation error correlation matrices
    C_LS = zeros(M,M,K,L);
    
end


%% Go through all cells
%for j = 1:L
%j=1;
%close all
%Go through all f pilot groups
for g = 1:f
    
    %Extract the cells that belong to pilot group g
    groupMembers = find(g==pilotPattern)';
    
    %Compute processed pilot signal for all UEs that use these pilots, according to (3.5)
    Htemp = H(:,:,:,g==pilotPattern);
    yp = sqrt(p)*tau_p*sum(Htemp,4) + sqrt(tau_p)*Np(:,:,:,g);
    
    %Go through all UEs
    for k = 1:K
        
        %Compute the matrix that is inverted in the MMSE estimator
        PsiInv = (p*tau_p*sum(R(:,:,k,g==pilotPattern),4) + eyeM);
        if rcond(PsiInv)<1e-15 %|| rcond(R(:,:,k,l))<1e-15
            clf
            pilotPattern
            rcond(PsiInv)
            rcond(R(:,:,k,l))
            imagesc(abs(((p*tau_p*sum(R(:,:,k,g==pilotPattern),4))))), colorbar
            drawnow
        end
        
        %If EW-MMSE estimation is to be computed
        if nargout >= 3
            %Compute a vector with elements that are inverted in the EW-MMSE estimator
            PsiInvDiag = diag(PsiInv);
        end
        
        %Go through the cells in pilot group g
        for l = groupMembers
            
            %Compute MMSE estimate of channel between BS l and UE k in
            %cell j using (3.9) in Theorem 3.1
            
            RPsi = R(:,:,k,l) / PsiInv;
            Hhat_MMSE(:,:,k,l) = sqrt(p)*RPsi*yp(:,:,k);
            
            %Compute corresponding estimation error correlation matrix, using (3.11)
            C_MMSE(:,:,k,l) = R(:,:,k,l) - p*tau_p*RPsi*R(:,:,k,l);
            
            
            %If EW-MMSE estimation is to be computed
            if nargout >= 3
                
                %Compute EW-MMSE estimate of channel between BS l and
                %UE k in cell j using (3.33)
                A_EW_MMSE = diag(sqrt(p)*diag(R(:,:,k,l)) ./ PsiInvDiag);
                Hhat_EW_MMSE(:,:,k,l) = A_EW_MMSE*yp(:,:,k);
                
                %Compute corresponding estimation error correlation
                %matrix, using the principle from (3.29)
                productAR = A_EW_MMSE * R(:,:,k,l);
                
                C_EW_MMSE(:,:,k,l) = R(:,:,k,l) - (productAR + productAR') * sqrt(p)*tau_p + tau_p*A_EW_MMSE*PsiInv*A_EW_MMSE';
                
            end
            
            
            %If LS estimation is to be computed
            if nargout >= 5
                
                %Compute LS estimate of channel between BS l and UE k
                %in cell j using (3.35) and (3.36)
                A_LS = 1/(sqrt(p)*tau_p);
                Hhat_LS(:,:,k,l) = A_LS*yp(:,:,k);
                
                %Compute corresponding estimation error correlation
                %matrix, using the principle from (3.29)
                productAR = A_LS * R(:,:,k,l);
                
                C_LS(:,:,k,l) = R(:,:,k,l) - (productAR + productAR') * sqrt(p)*tau_p + tau_p*A_LS*PsiInv*A_LS';
                
            end
            
        end
        
    end
    
end
