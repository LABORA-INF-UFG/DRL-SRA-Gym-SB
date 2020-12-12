function [noise] = gen_corr_complex_noise(covMat,nSamples)
% function [noise] = gen_corr_complex_noise(covMat,nSamples)

if isreal(covMat)  %assumes covMat is real
    
    noiseReal = gen_corr_noise(covMat, nSamples);
    noiseImag = gen_corr_noise(covMat, nSamples);
    
    %noiseReal
    %noise = sqrt(1/2) * complex(real(noiseReal), imag(noiseImag));
    noise = sqrt(1/2) * complex(noiseReal, noiseImag);
    
else
    noise = gen_corr_noise(covMat, nSamples);
end