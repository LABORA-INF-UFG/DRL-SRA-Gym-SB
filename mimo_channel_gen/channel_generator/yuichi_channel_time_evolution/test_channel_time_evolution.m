clearvars;
clc;

%% time
N = 1e5;

% covariance matrix
dimension = 4;
corrMat = randn(dimension) + 1i * randn(dimension);
R_bcu = corrMat * corrMat';

%noise generator
corrNoiseGen = CorrelatedComplexNoiseGenerator(R_bcu);

channelsOverTime = zeros(dimension, N); 

%initial channel at t = 1
chan = corrNoiseGen.generate(1);
channelsOverTime(:,1) = chan;

alpha = 0.5;
corrNoiseGen = CorrelatedComplexNoiseGenerator( (1 - alpha^2) * R_bcu);

for t=2:N
    channelsOverTime(:,t) = alpha * channelsOverTime(:, t-1) + corrNoiseGen.generate(1);
end

disp('estimated covariance matrix');
channelsOverTime * channelsOverTime' / N

disp('true covariance matrix');
R_bcu