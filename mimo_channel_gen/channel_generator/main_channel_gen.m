%% compared with hope2, this discards all intercell channels in stage 2 and saves much smaller files in end
%% compared with hope1 this one uses the ak_ita2019_episodes_creation_final.m
%% Compared with previous ak_ita2019_episodes_creation_different_alphas.m this version organizes the code
% and properly implements channel evolution of all intercell channels
% here we move step 1 of stage 1 to outside the loop that generates the
% channel episodes

%% This version stores in the mat the traffic (Poisson) and interference B(p) too

%% This version uses different alphas, to mimic LOS and NLOS users

%R             = M x M x K x L x L matrix with spatial correlation matrices
%                for all UEs in the network. R(:,:,k,j,l) is the correlation
%                matrix for the channel between UE k in cell j and the BS
%                in cell l. This matrix is normalized such that trace(R)=M.

%this version calculates C for all channels and estimates the
%interference for target cell
close all
clear *

rng(31);
%addpath('/MATLAB Drive/massivemimobook-master/Code','-end')
%addpath(fullfile(pwd, '/massivemimobook-master/Code'),'-end')
addpath('../massivemimobook-master/Code','-end')
addpath('../802_16_outdoor/genChannels')
output_folder = 'exp1/'; %end with /

run_time_evolution_analysis = 0; %plot and pause
run_SE_analysis = 1; %run analysis. Not important for the data to be generated for Python
%use_19_hexagonal = 1; %if 1, use hexagonal cells, if 0 use rectangular
load_pre_computed_files = 0; %if 1, simply load pre-computed files. Otherwise, compute R and gains

this_scenario = 1; %choose scenarios

if this_scenario  == 1 %use_19_hexagonal == 1
    target_cell_index=1; %target cell, 1 for L=19 and 6 for L=16
    L=19;
    frequency_reuse=3; %frequency reuse, adopt 1 in case of complete reuse or L for no reuse
    K=10; %# users
    M=64; %should read from PAR.bs_antenna_per_sector = 64;
    ASDdeg=25;
elseif this_scenario == 2
    target_cell_index=6; %target cell, 1 for L=19 and 6 for L=16
    L=16;
    frequency_reuse=2; %frequency reuse, adopt 1 in case of complete reuse or L for no reuse
    K=10; %# users
    M=64; %should read from PAR.bs_antenna_per_sector = 64;
    ASDdeg=25;
elseif this_scenario == 3
    target_cell_index=1; %target cell, 1 for L=19 and 6 for L=16
    L=7;
    frequency_reuse=1; %frequency reuse, adopt 1 in case of complete reuse or L for no reuse
    K=3; %# users
    M=8; %should read from PAR.bs_antenna_per_sector = 64;
    ASDdeg=10;
end

%Note that if M is large and ASDdeg small, the R matrices may be non positive
%definite


freqs = [24, 28];
gain_factors_dB = [0 8];

% RL episodes
num_episodes = 10;
num_blocks = 10; %5*20; %per episode
tau_p = 30;  %samples dedicated to pilots in coherence block
tau_u = 40;
tau_d = 140;
tau_c = tau_u + tau_p + tau_d; %samples in coherence block (samples per frame)
p=100; %Total uplink transmit power per UE (mW)
rho=100; %Total downlink transmit power per UE (mW)
%first half of users use alpha_los
fDTs_los=0.0001; %normalized Doppler shift >= 0.01
fDTs_nlos=0.001; %normalized Doppler shift >= 0.01
alpha_los = besselj(0,2*pi*fDTs_los); %0.9998; %AR coefficient to impose correlation over time
alpha_nlos = besselj(0,2*pi*fDTs_nlos); %AR coefficient to impose correlation over time
if 1
    nbrOfRealizationsForEstimation = 300;
else %in case wants that estimation and actual samples have same length
    nbrOfRealizationsForEstimation  = num_blocks*(tau_c-tau_p+1);
end
%Compute the prelog factor assuming only downlink transmission
if 1
    prelogFactor = tau_d/tau_c;
else
    prelogFactor = 160/190; %fixe it
end

if 0
    eval(['load ' output_folder '/all_locations.mat']) %load pre-computed BSpositions and UEpositions
    if 1
        close all
        ak_plot_cells_and_ues(UEpositions, BSpositions)
    end

    [K, L]=size(UEpositions);
end

%in practice the estimation is made with t_p samples, but we will
%assume 1 samples represents all of them (+1 below)
%we generate all slots, each with tau_c samples (or slots?)
Ne = num_blocks * tau_c; %num of samples in episode
num_total_samples = num_blocks * (tau_c - tau_p + 1); %old?
%num_total_samples=100; %number of channel realizations

[status, msg, msgID] = mkdir(output_folder);
[status, msg, msgID] = mkdir([output_folder 'channels/']);
addpath('./yuichi_channel_time_evolution','-end');
%addpath(fullfile(pwd, '/channel_generator/yuichi_channel_time_evolution'),'-end')
%addpath('yuichi_channel_time_evolution','-end');
%% Propagation parameters
BW=100e6; %100 MHz - Communication bandwidth
%Define noise figure at BS (in dB)
noiseFigure = 7;
%Compute noise power
noiseVariancedBm = -174 + 10*log10(BW) + noiseFigure;

accuracy=2; %1 or 2. 2 is faster but matrices may not be positive-definite

for freq=1:length(freqs)
    %% step 1 of stage 1. Here we use the full multicell processing, and we even
    %% estimate channels and SE for BSs other than the target BS

    output_file_name_prefix = [output_folder 'channels/channels_f' num2str(freq) '_b' num2str(num_blocks) 'tauc' num2str(tau_c) 'taup' num2str(tau_p)];

    if load_pre_computed_files == 1 %simply load pre-computed files
        eval(['load ' output_folder 'R_independent_channels_f' num2str(freq)])
        eval(['load ' output_folder 'channelGaindBOverNoise_f' num2str(freq)])
    else %calculate
        if this_scenario == 1 || this_scenario == 3
            ak_ita2019_generate_hexagoral_multi_cell_locations(L, output_folder)
            eval(['load ' output_folder 'all_locations']) %load BSpositions and UEpositions
            [R,channelGaindB] = ak_ita2019_functionExampleSetup4(BSpositions,UEpositions,M,ASDdeg,accuracy);
        else
            [R,channelGaindB] = functionExampleSetup(L,K,M,accuracy,ASDdeg);
        end
        %Compute the normalized average channel gain, where the normalization
        %is based on the noise power
        %channelGaindBOverNoise = channelGaindB - noiseVariancedBm;    %- gain_factors_dB(freq);
        channelGaindBOverNoise = channelGaindB - noiseVariancedBm - gain_factors_dB(freq);
        eval(['save -v6 ' output_folder 'R_independent_channels_f' num2str(freq) ' R']) %save R
        eval(['save -v6 ' output_folder 'channelGaindBOverNoise_f' num2str(freq) ' channelGaindBOverNoise']) %save channelGaindBOverNoise
    end

    %[R,channelGaindB] = ak_ita2019_functionExampleSetup2(BSpositions,UEpositions,M,ASDdeg); %only target cell
    %     if 1
    %         [R,channelGaindB] = ak_ita2019_functionExampleSetup3(BSpositions,UEpositions,M,ASDdeg,accuracy); %all cells
    %         channelGaindBOverNoise = channelGaindB - noiseVariancedBm - gain_factors_dB(freq);
    %         clear channelGaindB
    %         %re-scale with the given gain and eventually try to make R
    %         %positive-definite in case it is not
    %         R = ak_ita2019_scale_R_check_PD(R,channelGaindBOverNoise); %note noise is taken in account here
    %         eval(['save -v6 R_independent_channels_f' num2str(freq) ' R']) %save R
    %         eval(['save -v6 channelGaindBOverNoise_f' num2str(freq) ' channelGaindBOverNoise']) %save channelGaindBOverNoise
    %     else
    %         eval(['load R_independent_channels_f' num2str(freq)])
    %     end


    %note that the outuput R below is scaled by the channel gain (over noise), so
    %it differs from the input R
    %[Hhat_MMSE,C_MMSE,R,H,Hhat_EW_MMSE,C_EW_MMSE,Hhat_LS,C_LS]
    %[Hhat,C_MMSE,tau_p,R,H] = functionChannelEstimates(R,channelGaindBOverNoise,nbrOfRealizationsForEstimation,M,K,L,p,frequency_reuse);
    [Hhat,C_MMSE,R,H] = ak_functionChannelEstimates(R,channelGaindBOverNoise,nbrOfRealizationsForEstimation,M,K,L,p,frequency_reuse,tau_p);

    %[Hhat,C_MMSE,H] = ak_ita2019_channel_estimates_all_channels(R,nbrOfRealizationsForEstimation,p,frequency_reuse,tau_p);
    [M,~,K,L,~]=size(R);

    %AK-TODO: ak_intercell_interference must take in account the pilot
    %reuse factor. it`s not now
    [gamma_parameters, all_cells_intercell_interference, interferences] = ak_intercell_interference(H,Hhat,target_cell_index);
    target_cell_intercell_interference = all_cells_intercell_interference(:,target_cell_index); %keep only for the target cell

    [signal_MR,interf_MR] = functionComputeSINR_DL(H,Hhat,C_MMSE,tau_c,tau_p,nbrOfRealizationsForEstimation,M,K,L,p);

    SE_debug = functionComputeSE_DL_poweralloc(rho*ones(K,L),signal_MR,interf_MR,prelogFactor);

    clear H Hhat; %no need anymore, already estimated things with independent channels
    if 0 %have to enable due to MMSE precoding, in which C_MMSE is needed
        clear C_MMSE
    end

    %% Step 2 of stage 1: channel h realizations and estimations hhat
    %% We need to use all LK channels to the target cell in order to estimate
    %% channels taking in account the pilot contamination. We need both the
    %% LK R correlation matrices and also the LK channels. But after the
    %% channels are estimated, we keep here only the intra-target-cell K channels
    %% We save then only these K channels into files

    % We don't need the full 5-d R. Just the channels related to the
    % target cell, so we put them into R_target
    R_target = squeeze(R(:,:,:,:,target_cell_index));
    if 1
        clear R
    end
    Rsqrt_target = zeros(M,M,K,L);
    for u=1:K
        for c=1:L
            Rtemp = squeeze(R_target(:,:,u,c));
            [~,check_PD] = chol(Rtemp );
            if check_PD > 0
                warning('AK: Non positive definite')
                disp(['c = ' num2str(c) ' u= ' num2str(u)])
            end
            Rsqrt_target(:,:,u,c) = sqrtm(Rtemp);
        end
    end

    for e=1:num_episodes
        disp(['Processing frequency ' num2str(freqs(freq)) ' GHz, episode ' num2str(e) ' out of ' num2str(num_episodes)])
        
        %now sample the ones corresponding to start of coherence blocks
        %in practice the estimation is made with several samples, but we will
        %assume 1 represents all pilot samples dedicated to channel estimation
        %[Hhat_MMSE,C_MMSE,tau_p,H,Hhat_EW_MMSE,C_EW_MMSE,Hhat_LS,C_LS] = ak_ita2019_channel_estimates_all_channels(R,channelGaindBOverNoise,nbrOfRealizationsForEstimation,p,frequency_reuse);
        
        %generate actual realizations
        %H_target = ak_ita2019_channel_realizations(Rsqrt_target,R_target,num_total_samples,alpha);
        
        %Go through all channels and apply the channel gains to the spatial
        %correlation matrices
        %for j = 1:L
        %j = 1; %only for target cell
        %H_episode = zeros(M,Ne,K); %generate all Ne samples of episodes, for channels of all K UEs to the target BS
        H_episode = zeros(M,Ne,K,L); %generate all Ne samples of episodes, for channels of all K*L UEs to the target BS
        for l = 1:L
            for k = 1:K
                %create initial channel
                H_random = randn(M,1)+1i*randn(M,1);
                H_correlated = sqrt(0.5)*Rsqrt_target(:,:,k,l)*H_random;
                %now use AR model to generate other channels
                R_ue = R_target(:,:,k,l);
                if k <= K/2 %LOS users
                    H_episode(:,:,k,l) = evolve_channel_over_time(H_correlated,alpha_los,R_ue,Ne);
                else %NLOS users
                    H_episode(:,:,k,l) = evolve_channel_over_time(H_correlated,alpha_nlos,R_ue,Ne);
                end
            end
        end
        %extract a slot from the center (+1 to the right if tau_p is odd) of the pilot tones
        %if 1
        tau_p_center = ceil(tau_p/2);
        H_for_estimation = H_episode(:,tau_p_center:tau_c:end,:,:);
        %H_for_estimation = H_target(:,1:tau_c-tau_p+1:end,:,:,1);
        
        %[Hhat_MMSE,C_MMSE,Hhat_EW_MMSE,C_EW_MMSE,Hhat_LS,C_LS] = ak_ita2019_channel_estimates_only(R,H_for_estimation,num_blocks,p,frequency_reuse,tau_p);
        %R_target has info about all L*K channels to the target cell,
        %as well as H_for_estimation.
        % We need to use all LK R matrices and channels to the target cell in order to estimate
        % channels taking in account the pilot contamination. That is, we need both the
        % LK R correlation matrices and also the LK channels.
        %AK-TODO We will use this function that calculates all channels
        %and then discard the channels other than the target (waste calculation)
        %I am in a hurry and the calculation is intricate to distangle.
        Hhat_blocks = ak_ita2019_channel_estimates_only(R_target,H_for_estimation,num_blocks,p,frequency_reuse,tau_p);
        
        %% Keep only the estimated channel for the target cell
        Hhat_blocks_target = zeros(M,num_blocks,K);
        for b=1:num_blocks
            Hhat_blocks_target(:,b,:) = Hhat_blocks(:,b,:,target_cell_index);
        end
        
        %% Copy only the samples corresponding to DL blocks and to the target cell
        Hdownlink_target = zeros(M,num_blocks*tau_d,K);
        for b=0:num_blocks-1
            start_ndx_downlink = b*tau_d+1; %these indices are for the array that will be populated
            end_ndx_downlink = start_ndx_downlink + tau_d-1;
            %downlink channels, indices for the long array with all samples
            %in the episode
            start_ndx_in_episode = b*tau_c+1; %beginning of the b-th coherence block
            start_ndx_in_episode = start_ndx_in_episode + tau_p + tau_u; %skip pilots and uplink data
            end_ndx_in_episode = start_ndx_in_episode + tau_d-1; %duration of tau_d
            Hdownlink_target(:,start_ndx_downlink:end_ndx_downlink,:) = H_episode(:,start_ndx_in_episode:end_ndx_in_episode,:,target_cell_index);
        end
        
        output_file_name = [output_file_name_prefix 'e_' num2str(e) '.mat'];
        H = Hdownlink_target; %use only target cell channels
        Hhat = Hhat_blocks_target; %save only 1 estimated channel per block
        
        eval(['save -v6 ' output_file_name ' H Hhat'])
    end
    if run_SE_analysis == 1
        SEs_all_users = SE_debug; %AK-TODO need to check
        eval(['save -v6 ' output_folder 'SEs_all_users' num2str(freq) '.mat SEs_all_users'])
    end
    if run_time_evolution_analysis == 1
        %deprecated: evaluate_correlation_over_time(R_target, Hdownlink_target)
        evaluate_correlation_with_estimated_channel(R_target, Hdownlink_target, Hhat_blocks_target, target_cell_index)
        
        evaluate_correlation_over_space(Hdownlink_target);
        
    end
    
    eval(['save -v6 ' output_folder 'gamma_parameters' num2str(freq) '.mat gamma_parameters'])
end
