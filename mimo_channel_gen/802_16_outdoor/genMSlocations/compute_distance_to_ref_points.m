%
%   mMIMOsim: a Matlab simulator for massive MIMO systems created by Robert
%   W. Heath, Jr. (rheath@mimowireless.com) and Kien T. Truong
%   (kientruong@utexas.edu). 
%
%
%   compute the distances between a given point to a set of reference points
%   
%   input:
%   -   given_point: coordinates of the given point
%   -   ref_points: coordinates of the reference points
%   -   coord_type: type of input coordinates
%       +   1:  Cartesian coordinates
%       +   0:  polar coordinates
%
%   coordinate ordering in input parameters
%   -   Cartesian coordinates:
%       +   1st row is x coordinates
%       +   2nd row is y coordinates
%   -   polar coordinates
%       +   1st row is r (radius) coordinates
%       +   2nd row is theta (phase) coordinates
%
%   output:
%   -   distances: the distances from the given point to the reference points
%   
%   key expressions:
%   -   for polar coordinates
%       d = sqrt(r1^2 + r2^2 - 2 * r1 * r2 * cos(theta1 - theta2))
%   -   for Cartesian coordinates
%       d = sqrt((x1 - x2)^2 + (y1 - y2)^2)


function distances = ...
    compute_distance_to_ref_points(...
    given_point,...
    ref_points,...
    coord_type...
    )

%   default type of coordinates is Cartesian
if nargin < 3
    
    coord_type = 1;
    
end %   if(nargin)


if coord_type
    
    distances = sqrt((given_point(1) - ref_points(1,:)) .^ 2 +...
        (given_point(2) - ref_points(2,:)) .^ 2);
    
else
    
    distances = ...
        sqrt(given_point(1) ^ 2 + ref_points(1,:) .^ 2 - 2 * given_point(1)...
        * (ref_points(1,:) .* cos(given_point(2) - ref_points(2,:))));
    
end %   if(coord_type)

end %   compute_distance_to_ref_points()