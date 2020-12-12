%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   This program generates a given number of realizations of user locations
%   in a given network. 
%
%   -   The key input parameters are stored in PAR by calling set_parameters().
%
%   -   The output, which is the generated realizations of user locations
%       in the network, is stored in a data file with an appropriate name.

close all; 
%clear all; clc; format compact;
addpath('../mainFolder')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   initialize the structures containing the key parameters                    %                                                       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   initialize parameters by calling the function set_parameters without any
%   argument or scenario_idx = 0
PAR = set_parameters_outdoor_channels();

%   initialize the locations of wireless nodes in the network
LOC = initialize_location(PAR);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   generate key general parameters                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   generate base station locations
[LOC.cell_x, LOC.cell_y, LOC.bs_x, LOC.bs_y, LOC.bs_beam_orientation] =...
    generate_BS_location(PAR);

%   generate base station antenna cluster locations - [Note]: this works
%   only if we consider fixed antenna cluster locations. Need to move it into
%   the main loop if we consider average performance of many random antenna
%   cluster locations
LOC.bs_cluster_x = LOC.bs_x;

LOC.bs_cluster_y = LOC.bs_y;

[LOC.bs_cluster_r, LOC.bs_cluster_theta] =...
    cart2pol(LOC.bs_cluster_x, LOC.bs_cluster_y);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   reservation for the output parameters and key intermediate parameters      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   store locations of the users in the network in each realization
ms_x_loc = zeros(PAR.num_ms, PAR.num_ms_loc_realization);

ms_y_loc = ms_x_loc;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   main loop                                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   loop over different realizations of user locations
for ms_loc_realization_idx = 1:PAR.num_ms_loc_realization
    
    %   generate this user location realization for all users
    [ms_x_loc(:, ms_loc_realization_idx), ms_y_loc(:, ms_loc_realization_idx)] =...
        generate_ms_location(PAR, LOC);

end % for(ms_loc_realization_idx)

%== save current data

%   name of data file for storing the generated realizations of user locations
%filename = '20140828mMIMO01cluster07cell24user.mat';
filename = '../mainFolder/20140828mMIMO01cluster07cell20user.mat';

save(filename, 'ms_x_loc', 'ms_y_loc');
disp(['Wrote file ' filename])