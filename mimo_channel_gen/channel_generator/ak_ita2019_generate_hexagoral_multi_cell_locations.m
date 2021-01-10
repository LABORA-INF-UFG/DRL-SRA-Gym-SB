function ak_ita2019_generate_hexagoral_multi_cell_locations(L, output_folder)
%AK. Generate a single scenario of fixed wireless
%Note that I am using 802.11 models here

%close all; clear *; format compact;

%rng('default');
rng(129);

%% Choose experiment with L=7 or L=19, and exp2 or exp1
%L=7;
%output fileName
output_file_name = [output_folder 'all_locations.mat'];

if L==7
    use_aldebaro_alignment_of_two_users = 1;
else
    use_aldebaro_alignment_of_two_users = 0;
end
should_plot_UE_number = 1;


%edit set_parameters_outdoor_channels

%root_path = fullfile(pwd, '/')
root_path = fullfile('..', '/');

shouldPlot = 1;
numEpisodes = 1; %use 1 if it's for fixed wireless
numChannelRealizationsPerEpisode = 1;  %2500 leads to a 5 GB file and -v6 has a limit of 2 GB
addpath([root_path '802_16_outdoor/mainFolder'],'-end')
%addpath([root_path '802_11_indoor'],'-end')
%addpath([root_path 'Correlation_Multiple_Cluster'],'-end') %AK add to path
addpath([root_path '802_16_outdoor/genMSlocations'],'-end') %AK add to path
if shouldPlot == 1
    %plot cells
    addpath([root_path '802_16_outdoor/genChannels'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   initialize the structures containing the key parameters                    %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   initialize parameters by calling the function set_parameters without any
%   argument or scenario_idx = 0
%PAR = set_parameters_outdoor_channels();
if L==19
    PAR = ak_fixed_wireless_set_parameters_outdoor_channelsL19(); %for L=19
else
    PAR = ak_fixed_wireless_set_parameters_outdoor_channels(); %for L=7
end

if PAR.num_sector_per_cell ~= 1
    error('Was assuming that PAR.num_sector_per_cell = 1')
end

%   initialize the locations of wireless nodes in the network
LOC = initialize_location(PAR);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   set for defining the simulation scenario                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%-- number of antennas at a sector
%bs_antenna_per_sector_range = [100:100:900 1000:1000:30000];

%-- number of base statio antenna values
%num_bs_antenna_per_sector_val = length(bs_antenna_per_sector_range);

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

%   index of the coordinated sector cluster of interest
%coord_cluster_idx = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   main loop                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   generate this user location realization for all users
[ms_x_loc, ms_y_loc] = generate_ms_location(PAR, LOC);

%store all locations
allMSlocations = zeros(PAR.num_cell,PAR.num_ms_per_sector,2);
allBSlocations = zeros(PAR.num_cell,2); %the BS locations do not change
for ii=1:PAR.num_cell
    allBSlocations(ii,1) = LOC.bs_x(ii);
    allBSlocations(ii,2) = LOC.bs_y(ii);
end

if shouldPlot == 1
    figure(1)
    clf
    %plot cells
    plot_BS_locations(PAR);
    %run('../genChannels/test_generate_BS_location')
    title('Example of cells');
    hold on
end

%   get location of users corresponding to this user location realization
LOC.ms_x = ms_x_loc;
LOC.ms_y = ms_y_loc;

%organize the mobile stations into its BS's (X MS per BS)
%for saving
locX = reshape(LOC.ms_x,PAR.num_ms_per_sector,PAR.num_cell);
locY = reshape(LOC.ms_y,PAR.num_ms_per_sector,PAR.num_cell);
for kk=1:PAR.num_cell
    %allMSlocations = zeros(currentEpisode,PAR.num_cell,PAR.num_ms_per_sector,2);
    allMSlocations(kk,:,1) = locX(:,kk);
    allMSlocations(kk,:,2) = locY(:,kk);
end

if use_aldebaro_alignment_of_two_users == 1
    %AK - put 2 close to UE 3
    a=angle(allMSlocations(1,2,1)+1j*allMSlocations(1,2,2));
    delta=30;
    new_x=allMSlocations(1,2,1)+delta;
    new_y=tan(a)*new_x;
    allMSlocations(1,3,1) = new_x;
    allMSlocations(1,3,2) = new_y;
    %AK - put 2 of cell 2 close to UE 1 of cell 1
    allMSlocations(2,2,1) = allMSlocations(1,1,1)+15;
    allMSlocations(2,2,2) = allMSlocations(1,1,2)+15;
end

distances = ak_get_all_distances_MS_BS(PAR, LOC);
disp(['Min distance UE to its BS = ', num2str(min(distances(:))) ' meters'])

%organize the way we want
BSpositions = zeros(1,PAR.num_cell);
UEpositions = zeros(PAR.num_ms_per_sector, PAR.num_cell);
for i=1:PAR.num_cell
    BSpositions(i) = allBSlocations(i,1) + 1j*allBSlocations(i,2);
    for j=1:PAR.num_ms_per_sector
        %note the transpose:
        UEpositions(j,i) = allMSlocations(i,j,1) + 1j* allMSlocations(i,j,2);
    end
    %sort according to distance to BS
    this_UEpositions = UEpositions(:,i);
    distances = abs(this_UEpositions - BSpositions(i));
    [~,ordering] = sort(distances);
    UEpositions(:,i) = this_UEpositions(ordering);
end
clear allMSlocations

eval(['save ' output_file_name ' -v6 BSpositions UEpositions'])
disp(['Wrote ' output_file_name])

if shouldPlot == 1
    ak_plot_cells_and_ues(UEpositions, BSpositions);
end
