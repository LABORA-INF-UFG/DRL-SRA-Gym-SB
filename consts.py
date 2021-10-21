PACKET_SIZE_BITS = 1024
BUFFER_SIZE = 30 * 8 * PACKET_SIZE_BITS
MAX_PACKET_AGE = 30
MIN_REWARD = -100
MAX_REWARD = 100
MAX_SPECTRAL_EFF = 7.8
K = 10
F = [2,2]
F_D = "2-2"
#F = [2]
#F_D = "2-1"

BLOCKS_EP = 100

GAMMA = 0.90
GAMMA_D = '09'
#LR = 1e-03
LR = 0.007
LR_D = '007'
EPSILON = 1e-05
EPSILON_D = '1e-05'


### stationary channel
MODELS_FOLDER_STATIONARY = 'models/'

### non-stationary channel
MODELS_FOLDER = 'models_nstat/'

MODELS_MMW = 'models_mmw/'
MODELS_COLAB= 'models_colab/'
MODELS_FINAL= 'models_final/'
