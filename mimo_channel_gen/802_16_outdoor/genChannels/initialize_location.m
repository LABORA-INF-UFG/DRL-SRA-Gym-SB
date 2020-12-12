%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   Reserve memory for the fields of the structure storing the
%   location-related parameters
%   
%   Input:
%       -   parameters: the structure containing the values of the key
%       parameters of the simulation scenario
%   
%   Output:
%       -   location: the structure containing the locations of base
%       stations and users
%
%

function location = initialize_location(parameters)

%   reservation for the Cartesian coordinates of base stations
location.bs_x = zeros(parameters.num_bs_cluster, 1);

location.bs_y = location.bs_x;

location.cell_x = zeros(parameters.num_cell, 1);

location.cell_y = location.cell_x;

location.bs_beam_orientation = location.bs_x;

%   reservation for the Cartesian coordinates of users
location.ms_x = zeros(parameters.num_ms, 1);

location.ms_y = location.ms_x;

%   reservation for the polar coordinates of base station antenna clusters
location.bs_cluster_r = ...
    zeros(parameters.bs_cluster_per_coord_cluster, parameters.num_coord_cluster);

location.bs_cluster_theta = location.bs_cluster_r;

%   reservation for the Cartesian coordinates of base station antenna clusters
location.bs_cluster_x = location.bs_cluster_r;

location.bs_cluster_y = location.bs_cluster_r;

end % function initialize_location()