function r = makepsd(m, tol )
% function r = makepsd(m, tol )
%force matrix to be positive definite
%By Marcos Yuichi
if(nargin == 1)
    tol = 1e-15;
end
m = (m + m.')/2; %make symmetric

[eigenVectors,eigenValues] = eig(m);
eigenValues = diag(eigenValues);
eigenValues(eigenValues < 0) = tol;

r = eigenVectors * diag(eigenValues) * eigenVectors';

allEigPosititive = all(eig(r) >= 0);

[~, p] = chol(r);
choleskyPositive = (p == 0);

if ~(choleskyPositive && allEigPosititive)
    error('Resulting matrix is not positive definite: cholp=%d alleigp=%d', choleskyPositive, allEigPosititive);
end
