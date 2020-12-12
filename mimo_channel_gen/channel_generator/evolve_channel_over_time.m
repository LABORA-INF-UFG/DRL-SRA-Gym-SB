function channelsOverTime = evolve_channel_over_time(initialChannel,alpha,R_bcu,N)

%% N is time
% R_bcu is covariance matrix

[dimension, ~] = size(R_bcu);

channelsOverTime = zeros(dimension, N);

%initial channel at t = 1
%chan = transpose(gen_corr_complex_noise(R_bcu, 1));
channelsOverTime(:,1) = initialChannel;

%alpha = 0.5;

if 0 %old version, did not work with mvrdn for complex-numbers
    %function handle, currying of alpha and R_bcu
    getNextChannel = @(prevChan) channel_time_evolution(prevChan, alpha, R_bcu);
    
    for t=2:N
        channelsOverTime(:,t) = getNextChannel(channelsOverTime(:,t-1) );
    end
end

corrNoiseGen = CorrelatedComplexNoiseGenerator( (1 - alpha^2) * R_bcu);

for t=2:N
    channelsOverTime(:,t) = alpha * channelsOverTime(:, t-1) + corrNoiseGen.generate(1);
end