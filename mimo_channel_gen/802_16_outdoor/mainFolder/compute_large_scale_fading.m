%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%   
%   Compute pathlosses from a set of transmitters to a set of receivers
%   
%   Input:
%   -   PAR:
%
%   -   LOC:
%
%   -   freq_carrier: center of the frequency carrier in Hz
%   
%   Output:
%   -   large_scale_fading: matrix containing large scale fading for each
%   transmitter-receiver pair. This matrix has size of
%       +   number of transmitter locations
%       +   number of receiver locations
%
%
%   History:
%   
%   |Version|Date      |Change                               |Author      |
%   |       |          |                                     |            |
%
% AK: returns distance too

function [FDD_UL_large_scale_fading,...
    FDD_DL_large_scale_fading,...
    TDD_large_scale_fading, distance] = ...
    compute_large_scale_fading(PAR, LOC)

%   compute distances from the transmitter locations to the receiver locations

distance =...
    zeros(PAR.bs_cluster_per_coord_cluster, PAR.num_ms, PAR.num_coord_cluster);

for coord_cluster_idx = 1:PAR.num_coord_cluster
    
    x_diff = bsxfun(@minus, LOC.ms_x', LOC.bs_cluster_x(:, coord_cluster_idx));

    y_diff = bsxfun(@minus, LOC.ms_y', LOC.bs_cluster_y(:, coord_cluster_idx));
    
    [~, distance(:, :, coord_cluster_idx)] = cart2pol(x_diff, y_diff);
    
end % for(coord_cluster_idx)


%   compute the distance in kilometers
distance_km = distance / 1000;

%   compute large scale fading for each transmitter-receiver pair for each model
%   of large scale fading
switch PAR.pathloss_model_idx
        
    %   3GPP pathloss model 36.814 Rel-9 pp. 96, urban mirco, NLOS, Hexagonal
    %   cell layou: PL = 36.7log10(d) + 22.7 - 26log10(f_c), where d is in
    %   meters and f_c is in GHz, shadowing is not included
    case 0
        
        %   pathloss in dB
        FDD_UL_pathloss_dB =...
            36.7 * log10(distance_km) + 22.7 - 26 * log10(PAR.FDD_UL_freq_GHz);
        
        FDD_DL_pathloss_dB =...
            36.7 * log10(distance_km) + 22.7 - 26 * log10(PAR.FDD_DL_freq_GHz);
        
        TDD_pathloss_dB =...
            36.7 * log10(distance_km) + 22.7 - 26 * log10(PAR.TDD_freq_GHz);
        
        
        %   large scale fading, already represented in the form of losses
        FDD_UL_large_scale_fading =...
            10 .^ ((-1) * (FDD_UL_pathloss_dB + PAR.penetration_loss_dB) / 10);
        
        FDD_DL_large_scale_fading =...
            10 .^ ((-1) * (FDD_DL_pathloss_dB + PAR.penetration_loss_dB) / 10);
        
        TDD_large_scale_fading =...
            10 .^ ((-1) * (TDD_pathloss_dB + PAR.penetration_loss_dB) / 10);
                        
    otherwise
        
        errorMsg = 'This pathloss model is not supported yet!';
        error(errorMsg);
        
end % switch(pathloss_model_idx)

FDD_UL_large_scale_fading = squeeze(FDD_UL_large_scale_fading);

FDD_DL_large_scale_fading = squeeze(FDD_DL_large_scale_fading);

TDD_large_scale_fading = squeeze(TDD_large_scale_fading);

distance = squeeze(distance);

end % compute_large_scale_fading()