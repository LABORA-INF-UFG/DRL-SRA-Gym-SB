function [nextChannel] = channel_time_evolution(prevChannel, alpha, covMat)
%UNTITLED Summary of this function goes here
%   each channel is a column vector

[~, nSamples] = size(prevChannel); 

nextChannel = alpha * prevChannel + transpose(gen_corr_complex_noise((1 - alpha^2) * covMat, nSamples));

end

