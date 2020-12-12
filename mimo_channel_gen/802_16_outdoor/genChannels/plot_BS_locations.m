function plot_BS_locations(PAR)
%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%   Modified by Aldebaro
%   BRIEF DESCRIPTION: plot locations of base stations
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


%   INPUT

%   number of base stations in the network
num_bs = PAR.num_cell;


%   inter-site distance [m]
ISD = PAR.ISD;

%   cell type: 0 for hexagonal and 1 for circle
cell_type = PAR.sector_type;


%   OUTPUT

%   Cartesian coordinates of base stations
[bs_x, bs_y] = generate_BS_location_v2(num_bs, ISD, cell_type);



%   ILLUSTRATION
%figure;

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

half_cell_radius = cell_radius / 2;

half_ISD = 0.5 * ISD;

x_border =...
    [cell_radius, half_cell_radius, -half_cell_radius,...
    -cell_radius, -half_cell_radius, half_cell_radius, cell_radius];

y_border = [0, half_ISD, half_ISD, 0, -half_ISD, -half_ISD, 0];

x_center1 = [cell_radius, 0, -half_cell_radius];

y_center1 = [0, 0, half_ISD];

x_center2 = [0, -half_cell_radius];

y_center2 = [0, -half_ISD];

for bs_idx = 1:num_bs
    line(x_border + bs_x(bs_idx),...
        y_border + bs_y(bs_idx),'LineStyle','-'); hold on;
    line(x_center1 + bs_x(bs_idx),...
        y_center1 + bs_y(bs_idx),'LineStyle','--'); hold on;
    line(x_center2 + bs_x(bs_idx),...
        y_center2 + bs_y(bs_idx),'LineStyle','--'); hold on;
end % for(bs_idx)

plot(bs_x, bs_y, '^r', 'MarkerSize', 12);

