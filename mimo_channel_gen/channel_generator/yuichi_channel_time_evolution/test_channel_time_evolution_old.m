clearvars;
clc;

%% time
N = 1e5;

% covariance matrix
if 0 %real-valued
    R_bcu = [3 0.5 0.25; 0.5 2 0.5; 0.25 0.5 1];
else %complex-valued
    R_bcu = [3 0.5+1j 0.25-0.2j; 0.5-1j 2 0.4; 0.25+0.2j 0.4 1];
end

[dimension, ~] = size(R_bcu);

channelsOverTime = zeros(dimension, N);

%initial channel at t = 1
chan = transpose(gen_corr_complex_noise(R_bcu, 1));
channelsOverTime(:,1) = chan;

alpha = 0.5;

%function handle, currying of alpha and R_bcu
getNextChannel = @(prevChan) channel_time_evolution(prevChan, alpha, R_bcu);

for t=2:N
    channelsOverTime(:,t) = getNextChannel(channelsOverTime(:,t-1) );
end

disp('estimated covariance matrix');
%abs(channelsOverTime * channelsOverTime' / N)
conj(channelsOverTime * channelsOverTime' / N) %why the conjugate?

disp('true covariance matrix');
R_bcu