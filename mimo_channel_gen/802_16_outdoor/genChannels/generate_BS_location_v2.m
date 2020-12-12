%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   BRIEF DESCRIPTION: generate locations of base stations
%   
%
%   INPUT
%
%   -   num_bs: number of base stations in the network
%
%   -   ISD: inter-site distance [m]
%
%   -   cell_type: cell type
%   
%
%   OUTPUT
%
%   -   bs_x: vector of x coordinate of base stations
%
%   -   bs_y: vector of y coordinate of base stations


function [bs_x, bs_y] =...
    generate_BS_location(...
    num_bs,... number of base stations in the network
    ISD,... inter-site distance [m]
    cell_type... cell type
    )

%   reservation for polar coordinates of base stations
bs_x = zeros(num_bs, 1);

bs_y = bs_x;

%   network sectorization pattern
half_ISD = 0.5 * ISD;

one_ISD = ISD;

onehalf_ISD = 1.5 * ISD;

two_ISD = 2 * ISD;

twohalf_ISD = 2.5 * ISD;


%   radius of cells in meters
switch cell_type

    %   hexagonal cells
    case 0

        cell_radius = ISD / sqrt(3);

    %   circle cells
    case 1

        cell_radius = ISD / 2;

    otherwise

        errorMsg = 'This type of cells is not supported yet!';
        error(errorMsg);

end % switch(cell_type)


half_r = 0.5 * cell_radius;

one_r = cell_radius;

onehalf_r = 1.5 * cell_radius;
                
two_r = 2 * cell_radius;

twohalf_r = 2.5 * cell_radius;

three_r = 3 * cell_radius;

threehalf_r = 3.5 * cell_radius;

four_r = 4 * cell_radius;


%   polar coordinates of base stations [note] - although these codes do work, 
%   they can still be shortened.
switch cell_type
    
    case 0 % hexagonal
        
        switch num_bs
            
            case 1  %   no interference cells
                
                %   center cell of interest
                bs_x = 0;
                
                bs_y = 0;
                
            case 7  %   one layer of interference cells
                
                bs_x =...
                    [0, onehalf_r, 0, -onehalf_r,...
                    -onehalf_r, 0, onehalf_r];
                
                bs_y =...
                    [0, half_ISD, one_ISD, half_ISD,...
                    -half_ISD, -one_ISD, -half_ISD];
                
            case 19
                
                bs_x =...
                    [0,...
                    onehalf_r, 0, -onehalf_r,...
                    -onehalf_r, 0, onehalf_r,...
                    three_r, three_r, onehalf_r,...
                    0, -onehalf_r, -three_r,...
                    -three_r, -three_r, -onehalf_r,...
                    0, onehalf_r, three_r];
                
                bs_y =...
                    [0,...
                    half_ISD, one_ISD, half_ISD,...
                    -half_ISD, -one_ISD, -half_ISD,...
                    0,  one_ISD, onehalf_ISD,...
                    two_ISD, onehalf_ISD, one_ISD,...
                    0, -one_ISD, -onehalf_ISD,...
                    -two_ISD, -onehalf_ISD, -one_ISD]; 
                
             case 27
                
                bs_x = [one_r, -half_r, -half_r,...
                    -two_r, -two_r, -threehalf_r,...
                    -two_r, -threehalf_r, -threehalf_r,...
                    -half_r, -half_r, -two_r,...
                    twohalf_r, one_r, one_r,...
                    four_r, four_r, twohalf_r,...
                    four_r, twohalf_r, twohalf_r,...
                    one_r, one_r, -half_r,...
                    -half_r, -two_r, -threehalf_r];
                
                bs_y = ...
                    [0, half_ISD, -half_ISD,...
                    0, -one_ISD, -half_ISD,...
                    one_ISD, half_ISD, onehalf_ISD,...
                    twohalf_ISD, onehalf_ISD, two_ISD,...
                    onehalf_ISD, one_ISD, two_ISD,...
                    one_ISD, 0, half_ISD,...
                    -one_ISD, -onehalf_ISD, -half_ISD,...
                    -one_ISD, -two_ISD, -onehalf_ISD,...
                    -twohalf_ISD, -two_ISD, -onehalf_ISD];
                
            otherwise
                
                errorMsg = strcat('ERROR: This number of hexagonal cells',... 
                'is not supported yet!');
                error(errorMsg);
                
        end % switch (num_bs)
        
    case 1 % circle
        switch num_bs
            
            case 1  %   no inter-cell interference
                
                bs_x = 0;
                bs_y = 0;
                
            otherwise
                
                errorMsg = strcat('ERROR: This number of circle cells',...
                    'is not supported yet!');
                error(errorMsg);
                
        end % switch (num_bs)
        
    otherwise
        
        errorMsg = 'ERROR: This simulation scenario is not supported yet!';
        error(errorMsg);
        
end % switch(scenario_idx)

end % function generate_BS_location()