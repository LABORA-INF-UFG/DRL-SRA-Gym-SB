%Note that I am using 802.11 models here

close all; clear *; format compact;

%edit set_parameters_outdoor_channels

shouldPlot = 1;
numEpisodes = 1; %use 1 if it's for fixed wireless
numChannelRealizationsPerEpisode = 1;  %2500 leads to a 5 GB file and -v6 has a limit of 2 GB
addpath('../../802_11_indoor','-end')
addpath('../../Correlation_Multiple_Cluster','-end') %AK add to path
outputFileNamePrefix = 'allChannels'; %extension is mat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   initialize the structures containing the key parameters                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   initialize parameters by calling the function set_parameters without any
%   argument or scenario_idx = 0
PAR = set_parameters_outdoor_channels();

if PAR.num_sector_per_cell ~= 1
    error('Was assuming that PAR.num_sector_per_cell = 1')
end

if shouldPlot == 1
    %plot cells
    addpath('../genChannels')
end

%   initialize the locations of wireless nodes in the network
LOC = initialize_location(PAR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   set for defining the simulation scenario                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-- number of antennas at a sector
bs_antenna_per_sector_range = [100:100:900 1000:1000:30000];

%-- number of base statio antenna values
num_bs_antenna_per_sector_val = length(bs_antenna_per_sector_range);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   generate key general parameters                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   generate base station locations
[LOC.cell_x, LOC.cell_y, LOC.bs_x, LOC.bs_y, LOC.bs_beam_orientation] =...
    generate_BS_location(PAR);

%   generate base station antenna cluster locations - [KTT - note]: this works
%   only if we consider fixed antenna cluster locations. Need to move it into
%   the main loop if we consider average performance of many random antenna
%   cluster locations

LOC.bs_cluster_x =...
    reshape(LOC.bs_x,...
    PAR.bs_cluster_per_coord_cluster, PAR.num_coord_cluster);

LOC.bs_cluster_y =...
    reshape(LOC.bs_y,...
    PAR.bs_cluster_per_coord_cluster, PAR.num_coord_cluster);

[LOC.bs_cluster_r, LOC.bs_cluster_theta] =...
    cart2pol(LOC.bs_cluster_x, LOC.bs_cluster_y);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   reservation for the output parameters and key intermediate parameters      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%== parameters for experiments

start_ms_loc_realization_idx = 1;

end_ms_loc_realization_idx = numEpisodes; % num_ms_loc_realization;

if end_ms_loc_realization_idx > PAR.num_ms_loc_realization
    error(strcat('The maximum number of MS location realizations = ',...
        num2str(PAR.num_ms_loc_realization)));
end

expected_num_ms_loc_realization =...
    end_ms_loc_realization_idx - start_ms_loc_realization_idx + 1;

%   index of the coordinated sector cluster of interest
coord_cluster_idx = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   main loop                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%load file with locations of mobile users for many episodes
%or generate MS locations on-the-fly
locationsFileName = '20140828mMIMO01cluster07cell20user.mat';
if exist(locationsFileName, 'file')
    load(locationsFileName)
else
    run('..\genMSlocations\generate_MS_loc_realizations')
end

%store all locations
allMSlocations = zeros(numEpisodes,PAR.num_cell,PAR.num_ms_per_sector,2);
allBSlocations = zeros(PAR.num_cell,2); %the BS locations do not change
for ii=1:PAR.num_cell
    allBSlocations(ii,1) = LOC.bs_x(ii);
    allBSlocations(ii,2) = LOC.bs_y(ii);
end

currentEpisode = 0;
%   loop over different realizations of other user locations
for ms_loc_realization_idx =...
        start_ms_loc_realization_idx:end_ms_loc_realization_idx
    
    currentEpisode = currentEpisode + 1;
    allHs = []; %reset matrix that will store all episode
    
    if shouldPlot == 1
        figure(1)
        clf
        %plot cells
        plot_BS_locations(PAR);
        %run('../genChannels/test_generate_BS_location')
        title('Example of cells');
        hold on
    end
    
    if mod(ms_loc_realization_idx, 10) == 0
        %show_iteration = [ms_loc_realization_idx end_ms_loc_realization_idx]
        disp([num2str(ms_loc_realization_idx) ' / ' num2str(end_ms_loc_realization_idx)])
    end
    
    %   get location of users corresponding to this user location realization
    LOC.ms_x = ms_x_loc(:, ms_loc_realization_idx);
    LOC.ms_y = ms_y_loc(:, ms_loc_realization_idx);
    
    %organize the 140 mobile stations into its 7 BS's (20 MS per BS)
    %for saving
    locX = reshape(LOC.ms_x,PAR.num_ms_per_sector,PAR.num_cell);
    locY = reshape(LOC.ms_y,PAR.num_ms_per_sector,PAR.num_cell);
    for kk=1:PAR.num_cell
        %allMSlocations = zeros(currentEpisode,PAR.num_cell,PAR.num_ms_per_sector,2);
        allMSlocations(currentEpisode,kk,:,1) = locX(:,kk);
        allMSlocations(currentEpisode,kk,:,2) = locY(:,kk);
    end
    
    if shouldPlot == 1
        if exist('handlerPlot','var') %delete previous users
            delete(handlerPlot)
        end
        %handlerPlot = plot(locX, locY,'x');
        handlerPlot = plot(squeeze(allMSlocations(currentEpisode,:,:,1))', ...
            squeeze(allMSlocations(currentEpisode,:,:,2))','x');
        %plot(locX(1,:), locY(1,:),'o','MarkerSize',12)
        %title('BS and UE locations')
        %pause
        drawnow
    end
    
    %   compute large scale fading coefficents
    %Assume TDD
    %[FDD_UL_LSF, FDD_DL_LSF, TDD_LSF] = compute_large_scale_fading(PAR, LOC);
    %Dimension of TDD_LSF is 140 x 7 (all MS to all BS)
    [~, ~, TDD_LSF, distance] = compute_large_scale_fading(PAR, LOC);
    
    % Direction of connection
    Connection = 'downlink';
    for ii=1:size(distance,1)
        %handler_ms = plot(LOC.ms_x(ii), LOC.ms_y(ii),'o','MarkerSize',14);
        for jj=1:size(distance,2)
            
            disp(['MS = ' num2str(ii) ', BS = ', num2str(jj)])
            
            if exist('handler_line','var') %delete previous line
                delete(handler_line)
            end
            if 0 
            if shouldPlot == 1
                handler_line = line([LOC.ms_x(ii), LOC.bs_x(jj)], [LOC.ms_y(ii),LOC.bs_y(jj)]);
                %handler_bs = plot(LOC.bs_x(jj), LOC.bs_y(jj),'o','MarkerSize',14);
                drawnow
            end
            end
            
            % Distance Tx-Rx
            Distance_Tx_Rx_m = distance(ii,jj); %max is 30 m for case 'F'
            % Carrier frequency
            CarrierFrequency_Hz = PAR.TDD_freq;
            %get channel between MS ii and BS jj
            H = ak_getMIMOChannels(Connection, Distance_Tx_Rx_m, CarrierFrequency_Hz);
            
            if shouldPlot == 1
                figure(2)
                clf
                ak_plot_PDP(H,1,2)
                figure(3)
                numPath=1;
                pwelch(squeeze(H(1,2,numPath,:)))
                drawnow
                figure(1)
                %pause
            end
            
            %H has size (NumberOfTxAntennas,NumberOfRxAntennas,NumberOfPaths,NumberOfHSamples)
            [NumberOfTxAntennas,NumberOfRxAntennas,NumberOfPaths,NumberOfHSamples]=size(H);
            if numChannelRealizationsPerEpisode > NumberOfHSamples
                error('numChannelRealizationsPerEpisode > NumberOfHSamples')
            end
            %keep only the channels required by the user
            H = H(:,:,:,1:numChannelRealizationsPerEpisode);
            
            if ~exist('allHs','var')
                allHs = zeros(PAR.num_ms_per_sector,PAR.num_cell,NumberOfTxAntennas,NumberOfRxAntennas,NumberOfPaths,numChannelRealizationsPerEpisode);
            end
            allHs(ii,jj,:,:,:,:)=H;
            
        end
    end
    
    %Save as a mat to read with D:\gits\lasse\software\mimo-python\build\lib\mimopython\matlab_tofrom_python.py
    %that can read multidimensional complex-valued arrays saved in Matlab with something like:
    %save -v6 'test.mat'
    thisFileName = [outputFileNamePrefix '_' num2str(PAR.TDD_freq/1e9) 'GHz_Episode' num2str(currentEpisode) '.mat'];
    save(thisFileName,'allHs','allMSlocations','allBSlocations','-v6')
    %save(thisFileName,'allMSlocations','allBSlocations','-v6','-append')
    disp(['Wrote file ' thisFileName])
    
    
end % for(ms_loc_realization_idx)