%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   generate uniformly distributed locations in one cell relatively to location
%   of the base station at the center
%   
%   Input:
%       -   sector_type: type of cell geometry
%           +   0: hexagonal cells
%           +   1: circle cells
%       -   cell_radius: cell radius in meters
%       -   cell_hole_radius: radius of the ring centered at the base station
%       where users are prohibitive
%       -   num_ms_per_sector: number of users per cell
%       -   hole_loc_r: radius of center of circular holes where user are
%       prohibited in polar coordinates
%       -   hole_loc_theta: phase of center of circular holes where user
%       are prohibited in polar coordinates
%       -   hole_radius: radius of the prohibited circular holes
%   
%   Output:
%       -   location_one_sector_r: radius coordinate of locations of size
%       num_ms_per_sector
%
%       -   location_one_sector_phase: phase coordinate of locations of size
%       num_ms_per_sector
%
%
%   References
%
%   [1] M. Hlynka and D. Loach, "Generating uniform random points in a
%   regular n sided polygon", Windsor Math and Statistics Report, 2005,
%   available: http://web2.uwindsor.ca/math/hlynka/Ngon.pdf
%   
%   [2] C. Gorg and U. Vornefeld, "Uniform distribution of users in a
%   circle", available:
%   http://www.comnets.uni-bremen.de/itg/itgfg521/per_eval/circle_uniform_d
%   istribution.pdf

function [location_one_sector_r location_one_sector_theta] =...
    generate_one_sector_random_locations(...
    sector_type,...
    cell_radius,...
    num_ms_per_sector,...
    hole_loc_r,...     %   optional
    hole_loc_theta,... %   optional
    hole_radius...          %   optional
    )


%   radius of locations in one cell
location_one_sector_r = zeros(num_ms_per_sector, 1);

%   phase of locations in one cell
location_one_sector_theta = location_one_sector_r;

%   type of coordinates used computation
is_Cartesian_coordinates = 0;

%   combined polar coordinates of center of holes
hole_loc = [hole_loc_r; hole_loc_theta];


%   generate uniformly distributed random locations in a cell depending on the
%   type of cell
switch sector_type
    
    case 0 % hexagonal cells [1]
        
        %   generate uniform rv's in (0,1)
        u1 = rand(num_ms_per_sector, 1);
        
        u3 = rand(num_ms_per_sector, 1);
        
        %   number of gons
        n_gon = 6;
        
        %   the OAB angle in radian
        OAB_angle = (n_gon - 2) * pi / (2 * n_gon);
                
        %   compute the normalization factor
        k = cell_radius * sin(OAB_angle) * (...
            log(tan(OAB_angle / 2 + pi / n_gon)) -...
            log(tan(OAB_angle / 2)));
        
        %   generate the phase of user locations
        
        temp_theta = - OAB_angle +...
            2 * atan(exp((k * u1) / (cell_radius * sin(OAB_angle)) +...
            log(tan(OAB_angle / 2))));
        
        %   compute u for radius determination
        u =  floor(n_gon * u3);
        
        %   compute the phase of locations
        location_one_sector_theta = temp_theta +  u * 2 * pi / n_gon;
        
        %   compute R(temp_theta)
        R = cell_radius * sin(OAB_angle) ./ sin(temp_theta + OAB_angle);
        
        if nargin == 3
            %   generate uniform rv's in (0,1)
            u2 = rand(num_ms_per_sector, 1);
            
            %   compute the radius
            location_one_sector_r = R .* sqrt(u2);
        else
            %   generate the radius o locations, one-by-one to check
            %   the cell hole prohibition
            num_valid_loc = 1;
            
            while num_valid_loc <= num_ms_per_sector
                
                %   generate a candidate radius
                temp_loc_r = R(num_valid_loc) * sqrt(rand());
                
                temp_polar_loc =...
                    [temp_loc_r; location_one_sector_theta(num_valid_loc)];
                                
                %   check if it is outside the cell holes
                is_valid_loc =...
                    min(compute_distance_to_ref_points(temp_polar_loc,...
                    hole_loc, is_Cartesian_coordinates)) > hole_radius;

                %   if yes, add it to the location pool
                if is_valid_loc
                    
                    location_one_sector_r(num_valid_loc, 1) = temp_loc_r;
                    
                    num_valid_loc = num_valid_loc + 1;
                    
                end % if(is_valid_loc)
                
            end % while(num_valid_loc <= num_ms_per_sector)
        end % if(nargin)
        
    case 1 % circle cells [2]
        
        %   generate the phase of locatiopns
        location_one_sector_theta = 2 * pi * rand(num_ms_per_sector, 1);
        
        %   generate the radius o locations, one-by-one to check
        %   the cell hole prohibition
        num_valid_loc = 1;

        while num_valid_loc <= num_ms_per_sector
            
            %   generate a candidate radius
            temp_loc_r = cell_radius * sqrt(rand());
            
            temp_polar_loc = ...
                [temp_loc_r; location_one_sector_theta(num_valid_loc)];

            %   check if it is outside the cell hole
            is_valid_loc =...
                min(compute_distance_to_ref_points(temp_polar_loc,...
                hole_loc, is_Cartesian_coordinates)) > hole_radius;

            %   if yes, add it to the location pool
            if is_valid_loc
                
                location_one_sector_r(num_valid_loc, 1) = temp_loc_r;

                num_valid_loc = num_valid_loc + 1;
            end % if(is_valid_loc)
        end % while(num_valid_loc <= num_ms_per_sector)
        
    case 2 % diamond sectors in the 3-sector-per-cell pattern
        
        %   generate the radius o locations, one-by-one to check
        %   the cell hole prohibition
        num_valid_loc = 1;

        while num_valid_loc <= num_ms_per_sector
            
            %   generate a uniformly distributed location in the sector
            [temp_loc_r, temp_loc_theta] =...
                generate_a_location_in_a_diamond_sector(cell_radius);
            
            temp_polar_loc = [temp_loc_r; temp_loc_theta];

            %   check if it is outside the cell hole
            is_valid_loc =...
                min(compute_distance_to_ref_points(temp_polar_loc,...
                hole_loc, is_Cartesian_coordinates)) > hole_radius;

            %   if yes, add it to the location pool
            if is_valid_loc
                
                location_one_sector_r(num_valid_loc, 1) = temp_loc_r;
                
                location_one_sector_theta(num_valid_loc, 1) = temp_loc_theta;

                num_valid_loc = num_valid_loc + 1;
                
            end % if(is_valid_loc)
        end % while(num_valid_loc <= num_ms_per_sector)
        
    otherwise
        
        errorMsg = 'ERROR: This simulation scenario is not supported yet!';
        error(errorMsg);
        
end % switch(scenario_idx)

end % function generate_one_sector_random_location()