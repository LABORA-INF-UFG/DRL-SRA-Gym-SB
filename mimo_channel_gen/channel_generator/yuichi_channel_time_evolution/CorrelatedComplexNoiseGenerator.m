% new code from Yuichi, more robust to complex numbers
classdef CorrelatedComplexNoiseGenerator < handle
    properties
        corrMat
    end
    methods
        function obj = CorrelatedComplexNoiseGenerator(covMat)
            [v,d] = eig(covMat);
            obj.corrMat = v * sqrt(d);

        end
        function r = generate(obj, nSamples)
            [d, ~] = size(obj.corrMat);
            complexNoise1D = sqrt(.5) * randn(d * nSamples, 2) * [1; 1i];
            r = obj.corrMat * reshape(complexNoise1D, d, nSamples);
        end
    end
end