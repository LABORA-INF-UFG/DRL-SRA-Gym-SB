
rng(42);

N = 3;
nSamples = 1e6;
a = randn(N) + 1i * randn(N);
covMat = a * a';

noiseGen = CorrelatedComplexNoiseGenerator(covMat);
noise = noiseGen.generate(nSamples);

fprintf('true covMat:')
covMat

fprintf('estimated covMat:');
noise * noise' / nSamples
