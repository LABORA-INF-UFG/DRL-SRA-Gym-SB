%AK: get distances. Modification of compute_large_scale_fading.m
%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%   
%   Input:
%   -   PAR:
%
%   -   LOC:
%
%   Output:
%   -   distances
%
%   History:
%   
%   |Version|Date      |Change                               |Author      |
%   |       |          |                                     |            |
%

function distances = ak_get_all_distances_MS_BS(PAR, LOC)

%   compute distances from the transmitter locations to the receiver locations

distances =...
    zeros(PAR.bs_cluster_per_coord_cluster, PAR.num_ms, PAR.num_coord_cluster);

for coord_cluster_idx = 1:PAR.num_coord_cluster
    
    x_diff = bsxfun(@minus, LOC.ms_x', LOC.bs_cluster_x(:, coord_cluster_idx));

    y_diff = bsxfun(@minus, LOC.ms_y', LOC.bs_cluster_y(:, coord_cluster_idx));
    
    [~, distances(:, :, coord_cluster_idx)] = cart2pol(x_diff, y_diff);
    
end % for(coord_cluster_idx)

