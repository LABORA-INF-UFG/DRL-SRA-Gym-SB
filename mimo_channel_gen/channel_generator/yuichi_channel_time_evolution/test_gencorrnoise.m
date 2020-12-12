
covMat = [1 0.5 0.25; 0.5 1 0.5; 0.25 0.5 1];
nSamples = 1e6;
noise = gen_corr_noise(covMat, nSamples);

estimatedCovMat = noise' * noise / nSamples;

[m, n] = size(covMat);

MSE = sum(sum((covMat - estimatedCovMat).^2)) / (m*n);

fprintf('MSE = %f\n', MSE);