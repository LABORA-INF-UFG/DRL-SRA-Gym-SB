%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   generate locations of users in one cell relatively to location of the
%   base station at the center
%   
%   Input:
%       -   sector_type: type of cell geometry
%           +   0: hexagonal cells
%           +   1: circle cells
%       -   cell_radius: cell radius in meters
%       -   cell_hole_radius: radius of the ring centered at the base
%           station where users are prohibitive
%       -   num_ms_per_sector: number of users per cell
%   
%   Output:
%       -   ms_one_sector_x: vector of x coordinate of users of size
%       num_ms_per_sector
%       -   ms_one_sector_y: vector of y coordinate of users of size
%       num_ms_per_sector
%       -   hole_loc_r: radius of prohibited locations (holes) in the cell
%       -   hole_loc_theta: phase of prohibited locations (holes) in the
%       cell
%       -   hole_radius: radius of the prohibited circular holes
%
%   History:
%   
%   |Version|Date      |Change                               |Author      |
%   |       |          |                                     |            |
%   |1.0    |07/15/2012|Initial version                      |Kien Truong |


function [ms_one_sector_r ms_one_sector_theta] = ...
    generate_one_sector_users(...
    sector_type,...
    cell_radius,...
    num_ms_per_sector,...
    ms_loc_type,...
    hole_loc_r,...     %   optional
    hole_loc_theta,... %   optional
    hole_radius,...          %   optional
    ms_ring_radius...    %   optional
    )

%   radius of users in one sector
ms_one_sector_r = zeros(num_ms_per_sector, 1);

%   phase of users in one sector
ms_one_sector_theta = ms_one_sector_r;

%   type of coordinates used computation
is_Cartesian_coordinates = 0;

%   combined polar coordinates of center of holes
hole_loc = [hole_loc_r; hole_loc_theta];


%   generate locations of users in one cell depending on user location type
switch ms_loc_type
    
    
    %   equally distributed on a ring centered at the base station
    case 0
        
        %   set radius of user locations equal to the predetermined radius
        ms_one_sector_r = ms_ring_radius * ones(num_ms_per_sector, 1);
        
        %   compute the angular separation between two adjacent users
        ms_angular_separation = (2*pi) / num_ms_per_sector;
        
        %   set phases of user locations equally distributed from 0 to 2*pi
        start_idx = rand * ms_angular_separation;
        
        end_idx = start_idx + (num_ms_per_sector - 1) * ms_angular_separation;
        
        ms_one_sector_theta = start_idx:ms_angular_separation:end_idx;
        
        
    
    %   uniformly distributed on a ring
    case 1
        
        %   set radius of user locations equal to the predetermined radius
        ms_one_sector_r = ms_ring_radius * ones(num_ms_per_sector, 1);
        
        %   generate the phases of user locations
        num_valid_ms_loc = 1;
        
        %   loop until we have enough valid user locations
        while num_valid_ms_loc <= num_ms_per_sector
            
            %   generate a candidate user phase
            temp_ms_loc_theta = 2*pi * rand();
            
            %   compute the corresponding temporal user location
            temp_ms_loc = [ms_ring_radius, temp_ms_loc_theta];
            
            
            %   check if it is outside the cell holes
            is_valid_user_loc =...
                min(compute_distance_to_ref_points(temp_ms_loc,...
                hole_loc, is_Cartesian_coordinates)) > hole_radius;
            
            %   if yes, add it to the user location pool
            if is_valid_user_loc
                
                ms_one_sector_theta(num_valid_ms_loc, 1) = temp_ms_loc_theta;
                
                num_valid_ms_loc = num_valid_ms_loc + 1;
                
            end % if(is_valid_user_loc)
           
        end % while(num_valid_ms_loc <= num_ms_per_sector)
        
        
    %   uniformly distributed in the cell   
    case 2
    
        %   generate the polar coordinates of the users in one cell
        [ms_one_sector_r, ms_one_sector_theta] = ...
            generate_one_sector_random_locations(sector_type, cell_radius,...
            num_ms_per_sector, hole_loc_r, hole_loc_theta, hole_radius);
        
    otherwise
        
        errorMsg = strcat('This type of user location distribution is not',...
            ' currently supported yet!');
        error(errorMsg);
    
end % switch(ms_loc_type)


%   for checking the locations of the generated users in a diamond sector
% [ms_one_sector_x ms_one_sector_y] =...
%     pol2cart(ms_one_sector_theta, ms_one_sector_r);
% 
% figure;
% half_ISD = cell_radius * sqrt(3) / 2;
% half_cell_radius = cell_radius / 2;
% switch sector_type
%     case 0
%         x_border = [cell_radius, half_cell_radius, -half_cell_radius,...
%             -cell_radius, -half_cell_radius, half_cell_radius, cell_radius];
%         y_border = [0, half_ISD, half_ISD, 0, -half_ISD, -half_ISD, 0];
%         line(x_border, y_border), hold on;
%         plot(ms_one_sector_x, ms_one_sector_y, 'or');
%     case 2
%         x_border = ...
%             [cell_radius, half_cell_radius, -half_cell_radius, 0, cell_radius];
%         y_border = [0, half_ISD, half_ISD, 0, 0];
%         line(x_border, y_border), hold on;
%         plot(ms_one_sector_x, ms_one_sector_y, 'or');
%     otherwise
% 
%         errorMsg = 'ERROR: This simulation scenario is not supported yet!';
%         error(errorMsg);
% end % switch(sector_type)


end % function generate_one_sector_users()