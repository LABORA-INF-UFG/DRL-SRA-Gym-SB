%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   generate locations of users
%
%   
%   Input:
%       -   PAR: the structure containing the values of the key parameters of
%           the simulation scenario
%
%       -   LOC: the structur containing the values of the location-related
%           parameters
%   
%
%   Output:
%       -   ms_x: vector of x coordinate of users of size num_ms in relative of
%       the common origin of the whole network
%
%       -   ms_y: vector of y coordinate of users of size num_ms in relative of
%       the common origin of the whole network
%
%

function [ms_x, ms_y] = generate_ms_location(...
    PAR,...
    LOC...
    )

%   reservation for Cartesian coordinates of users
ms_x = zeros(PAR.num_ms, 1);

ms_y = ms_x;

%   reservation for Cartesian coordinates of users in one sector
ms_one_sector_x = zeros(PAR.num_ms_per_sector, 1);

ms_one_sector_y = ms_one_sector_x;

half_ISD = 0.5 * PAR.ISD;

onehalf_cell_radius = 1.5 * PAR.cell_radius;

ms_x2 = ms_x;
ms_y2 = ms_y;

%   generate user locations in cells, one by one
for bs_idx = 1:PAR.num_bs
    
    switch PAR.num_sector_per_cell
        case 1
            cell_idx = bs_idx;
            
        case 3
            cell_idx = floor((bs_idx - 1) / 3) + 1;
            
    end % switch(PAR.num_sector_per_cell)
    
    if (PAR.ms_loc_type == 0) || (PAR.ms_loc_type == 1)
    
        %   generate polar coordinates of user locations in the standard cell 
        [ms_one_sector_r, ms_one_sector_theta] = ...
            generate_one_sector_ms_location(PAR.sector_type, PAR.cell_radius,...
            PAR.num_ms_per_sector, PAR.ms_loc_type, LOC.bs_cluster_r,...
            LOC.bs_cluster_theta, PAR.hole_radius, PAR.ms_ring_radius);
    else
        
        %   generate polar coordinates of user locations in the standard cell 
        [ms_one_sector_r, ms_one_sector_theta] = ...
            generate_one_sector_ms_location(PAR.sector_type, PAR.cell_radius,...
            PAR.num_ms_per_sector, PAR.ms_loc_type, LOC.bs_cluster_r,...
            LOC.bs_cluster_theta, PAR.hole_radius);
        
    end % if(PAR.ms_loc_type)
    
    %   rotate the coordinates to the appropriate type of sectors
    switch PAR.num_sector_per_cell * 10 + PAR.coord_pattern;
        
        case {11, 12}
            ms_one_sector_theta =...
                ms_one_sector_theta + LOC.bs_beam_orientation(bs_idx);
        case {31, 32}
            ms_one_sector_theta =...
                ms_one_sector_theta...
                - mod((bs_idx - 1), PAR.num_sector_per_cell)...
                * (2 * pi / PAR.num_sector_per_cell);
    end % switch(PAR.coord_type)
            
            
    %   convert polar coordinates of user locations to Cartesian coordinates
    [ms_one_sector_x, ms_one_sector_y] =...
        pol2cart(ms_one_sector_theta, ms_one_sector_r);
    
    %   range of indices of users in the cell
    ms_start_idx = (bs_idx - 1) * PAR.num_ms_per_sector + 1;
    
    ms_end_idx = ms_start_idx + PAR.num_ms_per_sector - 1;
    
    %   translate user location from the standard cell to the cell
    ms_x(ms_start_idx:ms_end_idx) = LOC.cell_x(cell_idx) + ms_one_sector_x;
    
    ms_y(ms_start_idx:ms_end_idx) = LOC.cell_y(cell_idx) + ms_one_sector_y;
    
end % for(bs_idx)

%  for checking the locations of the generated users in the network
% figure;
% half_cell_radius = PAR.cell_radius / 2;
% 
% x_border =...
%     [PAR.cell_radius, half_cell_radius, -half_cell_radius,...
%     -PAR.cell_radius, -half_cell_radius, half_cell_radius, PAR.cell_radius];
% y_border = [0, half_ISD, half_ISD, 0, -half_ISD, -half_ISD, 0];
% x_center1 = [PAR.cell_radius, 0, -half_cell_radius];
% y_center1 = [0, 0, half_ISD];
% x_center2 = [0, -half_cell_radius];
% y_center2 = [0, -half_ISD];
% 
% 
% for cell_idx = 1:PAR.num_cell
%     line(x_border + LOC.cell_x(cell_idx),...
%         y_border + LOC.cell_y(cell_idx),'LineStyle','-'); hold on;
%     line(x_center1 + LOC.cell_x(cell_idx),...
%         y_center1 + LOC.cell_y(cell_idx),'LineStyle','--'); hold on;
%     line(x_center2 + LOC.cell_x(cell_idx),...
%         y_center2 + LOC.cell_y(cell_idx),'LineStyle','--'); hold on;
% end % for(bs_idx)
% 
% % %   for CoMP
% % plot(ms_x(1:3:81), ms_y(1:3:81), 'ok',...
% %     ms_x(2:3:81), ms_y(2:3:81), 'vk',...
% %     ms_x(3:3:81), ms_y(3:3:81), '+k',...
% %     LOC.cell_x, LOC.cell_y, '+r', LOC.bs_x, LOC.bs_y, 'sm')
% 
% %   for massive MIMO
% % plot(ms_x, ms_y, 'ok',...
% %     LOC.cell_x, LOC.cell_y, '+r', LOC.bs_x, LOC.bs_y, 'sm')


end % function generate_ms_location()