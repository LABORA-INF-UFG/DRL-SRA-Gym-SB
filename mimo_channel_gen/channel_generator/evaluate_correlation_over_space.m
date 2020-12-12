function evaluate_correlation_over_space(H)
L=size(H,4); %number of base stations (BS)

correlation_coeffs = ak_ita2019_calculate_correlation_coeff(H);
correlation_coeffs_average = squeeze(mean(correlation_coeffs,1));

figure
for i=2:L %first cell is target, just NaN
    clf
    imagesc(correlation_coeffs_average(:,:,i)), colorbar
    xlabel(['# UE at interferer cell = ' num2str(i)])
    ylabel('# UE at target cell = 1')
    %title(['Interferer cell = ' num2str(i)])
    pause
end

