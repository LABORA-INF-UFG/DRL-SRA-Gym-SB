% 2001-07-16 IEEE 802.16.3c-01/29r4
% understanding. This additional code is displayed in gray outlined boxes titled ‘Additional Info’.
% Let us first define the simulation parameters
N = 10000; %number of independent random realizations
OR = 4; %observation rate in Hz
M = 256; %number of taps of the Doppler filter
Dop_res = 0.1; %Doppler resolution of SUI parameter in Hz (used in resampling-process)
res_accu = 20; %accuracy of resampling process
%and the SUI channel parameters (SUI-3/omni used here):
P = [ 0 -5 -10 ]; %power in each tap in dB
K = [ 1 0 0 ]; %Ricean K-factor in linear scale
tau = [ 0.0 0.5 1.0 ]; %tap delay in µs
Dop = [ 0.4 0.4 0.4 ]; %Doppler maximal frequency parameter in Hz
ant_corr = 0.4; %antenna correlation (envelope correlation coefficient)
Fnorm = -1.5113; %gain normalization factor in dB
%First we calculate the power in the constant and random components of the Rice distribution for each tap:
P = 10.^(P/10); % calculate linear power
s2 = P./(K+1); % calculate variance
m2 = P.*(K./(K+1)); % calculate constant power
m = sqrt(m2); % calculate constant part
%Additional Info: RMS delay spread
rmsdel = sqrt( sum(P.*(tau.^2))/sum(P) - (sum(P.*tau)/sum(P))^2 );
fprintf('rms delay spread %6.3f µs\n', rmsdel);
%Now we can create the Ricean channel coefficients with the specified powers.
L = length(P); % number of taps
paths_r = sqrt(1/2)*(randn(L,N) + 1j*randn(L,N)).*((sqrt(s2))' * ones(1,N));
paths_c = m' * ones(1,N);
% Before combining the coefficient sets, the white spectrum is shaped according to the Doppler PSD function. Since the
% frequency-domain filtering function FFTFILT expects time-domain filter coefficients, we have to calculate these first.
% The filter is then normalized in time-domain.
for p = 1:L
    D = Dop(p) / max(Dop) / 2; % normalize to highest Doppler
    f0 = [0:M*D]/(M*D); % frequency vector
    PSD = 0.785*f0.^4 - 1.72*f0.^2 + 1.0; % PSD approximation
    filt = [ PSD(1:end-1) zeros(1,M-2*M*D) PSD(end:-1:2) ]; % S(f)
    filt = sqrt(filt); % from S(f) to |H(f)|
    filt = ifftshift(ifft(filt)); % get impulse response
    filt = real(filt); % want a real-valued filter
    filt = filt / sqrt(sum(filt.^2)); % normalize filter
    path = fftfilt(filt, [ paths_r(p,:) zeros(1,M) ]);
    paths_r(p,:) = path(1+M/2:end-M/2);
end
paths = paths_r + paths_c;
% Now that the fading channel is fully generated, we have to apply the normalization factor and, if applicable, the gain
% reduction factor
paths = paths * 10^(Fnorm/20); % multiply all coefficients with F
%Additional Info: average total tap power
Pest = mean(abs(paths).^2, 2);
fprintf('tap mean power level: %0.2f dB\n', 10*log10(Pest));
%Additional Info: spectral power distribution
%figure, pwelch(paths(1,:), 512, max(Dop));
%pwelch(paths(1,:))
figure(1), 
subplot(211),
ak_psd(paths(1,:),OR);
%xlabel('Frequency (Hz)')
ylabel('PSD (dB)');
%figure(2), freqz(filt);
subplot(212),
[H,w]=freqz(filt);
%figure(3)
plot(w/pi*(OR/2),20*log10(abs(H)))
ylabel('|H(f)|^2 (dB)')
xlabel('Frequency (Hz)')
grid

% In a multichannel scenario, the taps between different channels have to be correlated according to the specified antenna
% correlation coefficient. Assuming that two random channels have been generated in paths1 = paths_r1 +
% paths_c1 and paths2 = paths_r2 + paths_c2 following the procedures above, we can now apply the correlation

return

rho = ant_corr; % desired correlation is ant_corr
R = sqrtm([ 1 0 rho rho ; ... % factored correlation matrix
    0 1 rho rho ; ...
    rho rho 1 0 ; ...
    rho rho 0 1 ]);
V = zeros(4,L,N);
V(1,:,:) = real( paths_r1 ); % split complex coefficients
V(2,:,:) = imag( paths_r1 );
V(3,:,:) = real( paths_r2 );
V(4,:,:) = imag( paths_r2 );
for l = 1:L
    V(:,l,:) = R*squeeze(V(:,l,:)); % transformation
end
paths_r1 = squeeze(V(1,:,:)) + j*squeeze(V(2,:,:)); % combine complex coefficients
paths_r2 = squeeze(V(3,:,:)) + j*squeeze(V(4,:,:));
paths1 = paths_r1 + paths_c1; % add mean/constant part
paths2 = paths_r2 + paths_c2;
% Additional Info: estimate envelope correlation coefficient
% disp('estimated envelope correlation coefficients between all taps in both paths');
% disp('matrix rows/columns: path1: tap1, tap2, tap3, path2: tap1, tap2, tap3');
abs(corrcoef([ paths1; paths2]'));
% Finally, we resample the current rate to the specified observation rate. In order to use the Matlab polyphase
% implementation resample, we need the resampling factor F specified as a fraction F = P/Q.
SR = max(Dop)*2; % implicit sample rate
m = lcm(SR/Dop_res, OR/Dop_res);
P = m/SR*Dop_res; % find nominator
Q = m/OR*Dop_res; % find denominator
paths_OR = zeros(L,ceil(N*P/Q)); % create new array
for p=1:L
    paths_OR(p,:) = resample(paths(p,:), P, Q, res_accu);
end
% The resampled set of channel coefficients for all the 3 taps are now contained in the matrix paths_OR. The total
% simulated observation period of the channel is now SR·N = OR·ceil(N·P/Q), where SR = 2·max(Dop).

disp('AK: Need to organize code that generates independent coeff into a function, invoke 4 times and then correlate them')