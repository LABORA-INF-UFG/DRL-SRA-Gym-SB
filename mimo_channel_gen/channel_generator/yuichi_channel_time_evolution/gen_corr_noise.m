function [noise] = gen_corr_noise(cov_mat,n_samples)
%GENCORRNOISE Generates correlated noise
%   Generates a matrix where each row is a vector with noise with
%correlation corresponding to the cov_mat covariance matrix.

[n_dim, ~] = size(cov_mat);

mu = zeros(n_dim,1); %zero mean

noise = mvnrnd(mu, cov_mat, n_samples);

end

