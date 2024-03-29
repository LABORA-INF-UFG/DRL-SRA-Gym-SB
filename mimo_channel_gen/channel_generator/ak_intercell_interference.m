function [gamma_parameters, all_cells_intercell_interference, interferences] = ak_intercell_interference(H,Hhat,target_cell_index)
%not returning [signal_values,,average_signal, average_interf]
%% AK: get the individual realizations for signal and interference for MR
%consider only the inter-cell interference
%Compute DL SE for different transmit precoding schemes using Theorem 4.6.
%
%INPUT:
%H                 = M x nbrOfRealizations x K x L x L matrix with the
%                    exact channel realizations
%Hhat              = M x nbrOfRealizations x K x L x L matrix with the MMSE
%                    channel estimates
%C                 = M x M x K x L x L matrix with estimation error
%                    correlation matrices when using MMSE estimation.
%tau_c             = Length of coherence block
%tau_p             = Length of pilot sequences
%nbrOfRealizations = Number of channel realizations
%M                 = Number of antennas per BS
%K                 = Number of UEs per cell
%L                 = Number of BSs and cells
%p                 = Uplink transmit power per UE (same for everyone)
%
%OUTPUT:
%signal_MR    = K x L matrix where element (k,j) is a_jk in (7.2) with MR
%               precoding
%interf_MR    = K x L x K x L matrix where (l,i,j,k) is b_lijk in (7.3)
%               with MR precoding
%l is the interferer cell
%i is the UE in cell l
%j is the BS
%k is the user served by j
%signal_RZF   = Same as signal_MR but with RZF precoding
%interf_RZF   = Same as interf_MR but with RZF precoding
%signal_MMMSE = Same as signal_MR but with M-MMSE precoding
%interf_MMMSE = Same as interf_MR but with M-MMSE precoding
%prelogFactor = Prelog factor
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

[M,nbrOfRealizations,K,L,~] = size(H);

%Store identity matrices of different sizes
%eyeK = eye(K);
%eyeM = eye(M);

%intercell_interference_values(a,b,c,d,e) is the intercell interference
%that the BS provokes into user c in cell d when transmitting to user a in
%cell b. And e are the realizations
intercell_interference_values = zeros(K,L,K,L,nbrOfRealizations);
%Prepare to store simulation results for interference powers
average_interf = zeros(K,L,K,L);

%% Go through all channel realizations
for n = 1:nbrOfRealizations
    %if rem(n,100)==0
    %    n%AK
    %end
    %Go through all cells, because they will all receive interference from
    %the BS j
    for j = 1:L
        if 1 %use only intrachannel
            %faster method
            Hallj = reshape(H(:,n,:,:,j),[M K*L]);
            %effective channels H are made zero
            Hallj(:,K*(j-1)+1:K*j) = zeros(M,K); %zero intracell, to keep only intercell
        else %use all channels, inclusive intrachannel
            Hallj = reshape(H(:,n,:,:,j),[M K*L]);
        end
        %Extract channel realizations from all UEs to BS j
        Hhatallj = reshape(Hhat(:,n,:,:,j),[M K*L]);
        
        %Compute MR combining in (4.11), these precoders will influence the
        %DL signal of BS and will cause interference in the other cells
        V_MR = Hhatallj(:,K*(j-1)+1:K*j); %from estimated channels
        
        
        %Go through all UEs in cell j
        for k = 1:K
            
            %if j==13 && k==5
            %    disp('oo')
            %end
            
            if norm(V_MR(:,k))>0
                
                %%MR precoding
                w = V_MR(:,k)/norm(V_MR(:,k)); %Extract precoding vector for user k and make it norm=1
                                
                %original
                if 1
                    new_interf_parcel = reshape(abs(w'*Hallj).^2,[1 1 K L]); %precoding from BS j to all UEs, including the ones at cell j and all other cells
                else
                    %new, no abs
                    new_interf_parcel = reshape(w'*Hallj,[1 1 K L]); %precoding from BS j to all UEs, including the ones at cell j and all other cells
                end
                
                average_interf(k,j,:,:) = average_interf(k,j,:,:) + new_interf_parcel/nbrOfRealizations;
                intercell_interference_values(k,j,:,:,n) = new_interf_parcel;
                %interference_values = K x L x K x L x n matrix where (l,i,j,k) is b_lijk in (7.3)
                %               with MR precoding
                %l is the interferer cell
                %i is the UE in cell l
                %j is the BS
                %k is the user served by j
                
            end
            
        end
        
    end
    
end


interferences = squeeze(sum(sum(intercell_interference_values,2),1));

%get the distributions
gamma_parameters = zeros(K,2);
for u=1:K
    %for target cell:
    pd = fitdist(squeeze(interferences(u,target_cell_index,:)),'gamma');
    gamma_parameters(u,1) = pd.a;
    gamma_parameters(u,2) = pd.b;
end

%average interference
all_cells_intercell_interference = squeeze(sum(interferences,3)) / nbrOfRealizations;

if 0
    x=squeeze(intercell_interference_values(1,1,:,:,:));
    x=squeeze(sum(sum(x,2),1));
    %histfit(x,100,'rayleigh')
    histfit(x,100,'gamma')
    xlabel('\emph{I_{bu}^{inter}}')
    ylabel('Histogram count')
    
    %histfit(x,100,'exponential')
    
    histfit(squeeze(real(intercell_interference_values(1,1,6,3,:))),100,'exponential')
    xlabel('Power (mW)')
    ylabel('Histogram count')
end

