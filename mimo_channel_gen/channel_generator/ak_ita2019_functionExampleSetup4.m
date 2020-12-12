function [R,channelGaindB] = ak_ita2019_functionExampleSetup4(BSpositions,UEpositions,M,ASDdeg,accuracy)
%% This one does not normalize here. It normalizes outside
% AK: this function computes gain for all L^2 K UEs. It also disables wrapping given that I am using hexagons not rectangles
%the previous one assumed the target cell is the first 1 and saves computations with
%respect to other version
%Using to generate channel statistiscs

%addpath('/MATLAB Drive/massivemimobook-master/Code','-end')
addpath(fullfile(pwd, '/massivemimobook-master/Code'),'-end')

%rng(15);

L=length(BSpositions);
K=size(UEpositions,1);
%M=2; %should read from PAR.bs_antenna_per_sector = 64;
%This function generates the channel statistics between UEs at random
%locations and the BSs in the running example, defined in Section 4.1.3.
%
%Note that all distances in this code are measured in meters, while
%distances is often measured in kilometer in the monograph.
%
%INPUT:
%L        = Number of BSs and cells
%K        = Number of UEs per cell
%M        = Number of antennas per BS
%accuracy = Compute exact correlation matrices from the local scattering
%           model if approx=1. Compute a small-angle approximation of the
%           model if approx=2
%ASDdeg   = Angular standard deviation around the nominal angle
%           (measured in degrees)
%
%OUTPUT:
%R             = M x M x K x L x L matrix with spatial correlation matrices
%                for all UEs in the network. R(:,:,k,j,l) is the correlation
%                matrix for the channel between UE k in cell j and the BS
%                in cell l. This matrix is normalized such that trace(R)=M.
%channelGaindB = K x L x L matrix containing the average channel gains in
%                dB of all the channels. The product
%                R(:,:,k,j,l)*10^(channelGaindB(k,j,l)/10) is the full
%                spatial channel correlation matrix.
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


%% Model parameters

if 1
    %Pathloss exponent
    alpha = 3.76;
    
    %Average channel gain in dB at a reference distance of 1 meter. Note that
    %-35.3 dB corresponds to -148.1 dB at 1 km, using pathloss exponent 3.76
    constantTerm = -35.3;
else
    %Pathloss exponent
    alpha = 6;
    
    %Average channel gain in dB at a reference distance of 1 meter. Note that
    %-35.3 dB corresponds to -148.1 dB at 1 km, using pathloss exponent 3.76
    constantTerm = -15.3;
end

%Standard deviation of shadow fading
sigma_sf = 8;

%Minimum distance between BSs and UEs
%minDistance = 35;

%Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2; %Half wavelength distance

if 0
    %Distance between BSs in vertical/horizontal direction
    interBSDistance = abs(BSpositions(1)-BSpositions(2));
    squareLength = interBSDistance*4; %AK - work around, fix it later
    
    %Compute all nine alternatives of the BS locations when using wrap around
    wrapHorizontal = repmat([-squareLength 0 squareLength],[3 1]);
    wrapVertical = wrapHorizontal';
    wrapLocations = wrapHorizontal(:)' + 1i*wrapVertical(:)';
    BSpositions=transpose(BSpositions); %AK
    BSpositionsWrapped = repmat(BSpositions,[1 length(wrapLocations)]) + repmat(wrapLocations,[L 1]);
end

%Prepare to store normalized spatial correlation matrices
%R = zeros(M,M,K,L,L,length(ASDdeg));
%R = zeros(K,L,L,M,M);
R = zeros(M,M,K,L,L);

%Prepare to store average channel gain numbers (in dB)
channelGaindB = zeros(K,L,L);


%% Go through all the cells
for l = 1:L
    %disp(['Cell ' num2str(l)])
    
    %Go through all BSs
    for j = 1:L
        %disp(['BS ' num2str(j)])
        %Compute the distance from the UEs in cell l to BS j with a wrap
        %around topology, where the shortest distance between a UE and the
        %nine different locations of a BS is considered
        %[distancesBSj,whichpos] = min(abs( repmat(UEpositions(:,l),[1 size(BSpositionsWrapped,2)]) - repmat(BSpositionsWrapped(j,:),[K 1]) ),[],2);
        distancesBSj = abs(UEpositions(:,l) - BSpositions(j));
        
        %Compute average channel gain using the large-scale fading model in
        %(2.3), while neglecting the shadow fading
        channelGaindB(:,l,j) = constantTerm - alpha*10*log10(distancesBSj);
        
        %Compute nominal angles between UE k in cell l and BS j, and
        %generate spatial correlation matrices for the channels using the
        %local scattering model
        for k = 1:K
            
            %angleBSj = angle(UEpositions(k,l)-BSpositionsWrapped(j,whichpos(k)));
            angleBSj = angle(UEpositions(k,l) - BSpositions(j));
            
            if accuracy == 1 %Use the exact implementation of the local scattering model
                
                %R(k,l,j,:,:) = functionRlocalscattering(M,angleBSj,ASDdeg,antennaSpacing);
                R(:,:,k,l,j) = functionRlocalscattering(M,angleBSj,ASDdeg,antennaSpacing);
                
            elseif accuracy == 2 %Use the approximate implementation of the local scattering model
                
                R(:,:,k,l,j) = functionRlocalscatteringApprox(M,angleBSj,ASDdeg,antennaSpacing);
                %R(k,l,j,:,:) = functionRlocalscatteringApprox(M,angleBSj,ASDdeg,antennaSpacing);
                
            end
            
        end
        
    end
    
    %AK-TODO instead of generating L and then checking if above maximum,
    %better to generate each sample and test. For that must generate first
    %the one corresponding to BS (the one that should be the max)
    %Go through all UEs in cell l and generate shadow fading realizations
    for k = 1:K
        %disp(['UE ' num2str(k)])
        %Generate shadow fading realizations
        shadowing = sigma_sf*randn(1,1,L);
        channelGainShadowing = channelGaindB(k,l,:) + shadowing;
        
        %Check if another BS has a larger average channel gain to the UE
        %than BS l
        attempts = 0;
        while channelGainShadowing(l) < max(channelGainShadowing)
            
            %Generate new shadow fading realizations (until all UEs in cell
            %l has its largest average channel gain from BS l)
            shadowing = sigma_sf*randn(1,1,L);
            channelGainShadowing = channelGaindB(k,l,:) + shadowing;
            %AK: will avoid the infinite loop that I am facing:
            attempts = attempts + 1;
            if attempts > 10
                warning('Fixing shadowing to make sure each UE has the best signal from its own cell BS')
                indices_of_large_gains = find(channelGainShadowing > channelGainShadowing(l));
                discount_in_dB = 1;
                channelGainShadowing(indices_of_large_gains) = channelGainShadowing(l) - discount_in_dB;
                disp(['BS channelGainShadowing=' num2str(channelGainShadowing(l)) ' and new value is = ' num2str(max(channelGainShadowing)-discount_in_dB) ])
                break
            end
        end
        
        %Store average channel gains with shadowing fading
        channelGaindB(k,l,:) = channelGainShadowing;
        
    end
    
end

