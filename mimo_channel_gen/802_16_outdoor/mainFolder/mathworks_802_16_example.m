%From
%https://www.mathworks.com/help/comm/examples/ieee-802-16-channel-models.html

close all

rng default                 % Set random number generator for repeatability
M = 2;                      % Modulation order
Rsym = 10e3;                % Input symbol rate
Rbit = Rsym*log2(M);        % Input bit rate
Nos = 4;                    % Oversampling factor
Rs = Rbit*Nos;              % Input sample rate

tau = [0 0.4 0.9]*1e-6;    % Path delays, in seconds
pdb = [0 -15 -20];         % Average path gains, in dB
fd  = 0.5;                 % Maximum Doppler shift for all paths (identical)
ds  = doppler('Rounded');  % Rounded Doppler spectrum
Nt  = 2;                   % Number of transmit antennas
Nr  = 1;                   % Number of receive antennas
Rt  = toeplitz([1 0.7]);   % Transmit correlation matrix with correlation coefficient 0.7
kf  = 4;                   % Rician K-factor on the first path

chan = comm.MIMOChannel( ...
    'SampleRate', Rs, ...
    'PathDelays', tau, ...
    'AveragePathGains', pdb, ...
    'MaximumDopplerShift', fd, ...
    'DopplerSpectrum', ds, ...
    'TransmitCorrelationMatrix', Rt, ...
    'ReceiveCorrelationMatrix', 1, ...
    'FadingDistribution', 'Rician', ...
    'KFactor', kf, ...
    'PathGainsOutputPort', true);

Ns     = 1.5e6;      % Total number of channel samples
frmLen = 1e3;        % Number of samples per frame
numFrm = Ns/frmLen;  % Number of frames

chanOut   = zeros(Ns, Nr);
pathGains = zeros(Ns, length(tau), Nt);
for frmIdx = 1:numFrm
    inputSig = pskmod(randi([0 M-1], frmLen, Nt), M);
    idx = (frmIdx-1)*frmLen + (1:frmLen);
    [chanOut(idx,:), pathGains(idx,:,:)] = chan(inputSig);
end


figure;
win = hamming(Ns/5);
Noverlap = Ns/10;
pwelch(pathGains(:,2,1),win,Noverlap,[],Rs,'centered')
axis([-0.1/10 0.1/10 -80 10]);
legend('Simulation');


f  = -fd:0.01:fd;
a  = ds.Polynomial;      % Parameters of the rounded Doppler spectrum
Sd = 1/(2*fd*(a(1)+a(2)/3+a(3)/5))*(a(1)+a(2)*(f/fd).^2+a(3)*(f/fd).^4);
Sd = Sd*10^(pdb(2)/10);  % Scaling by average path power

hold on;
plot(f(Sd>0)/1e3,10*log10(Sd(Sd>0)),'k--');
legend('Simulation','Theory');


figure;
pwelch(pathGains(:,2,2),win,Noverlap,[],Rs,'centered')
axis([-0.1/10 0.1/10 -80 10]);
legend('Simulation');
hold on;
plot(f(Sd>0)/1e3,10*log10(Sd(Sd>0)),'k--');
legend('Simulation','Theory');

figure;
semilogy(abs(pathGains(:,1,1)),'b');
hold on;
grid on;
semilogy(abs(pathGains(:,1,2)),'r');
legend('First transmit link','Second transmit link');
title('Fading envelopes for two transmit links of Path 1');

figure;
semilogy(abs(pathGains(:,2,1)),'b');
hold on;
grid on;
semilogy(abs(pathGains(:,2,2)),'r');
legend('First transmit link','Second transmit link');
title('Fading envelopes for two transmit links of Path 2');

figure;
semilogy(abs(pathGains(:,3,1)),'b');
hold on;
grid on;
semilogy(abs(pathGains(:,3,2)),'r');
legend('First transmit link','Second transmit link');
title('Fading envelopes for two transmit links of Path 3');


TxCorrMatrixPath1 = corrcoef(pathGains(:,1,1),pathGains(:,1,2)).'
TxCorrMatrixPath2 = corrcoef(pathGains(:,2,1),pathGains(:,2,2)).'
TxCorrMatrixPath3 = corrcoef(pathGains(:,3,1),pathGains(:,3,2)).'