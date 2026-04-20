import numpy as np

# ============================================
# PREPROCESS GEOMETRY CONSTANTS
# ============================================

R_X = 150

R1 = 1000
R2 = 1140

H_PX = 1300

SCALE = np.float32(37294.15914879467)
OFFSET_Z = np.float32(-0.3/100)

NX = R_X
NZ = 2048

MIN_MEAN = 1061
CENTER_X = MIN_MEAN
BORDER_X = CENTER_X + R_X

# ============================================
# DATASET NORMALIZATION CONSTANTS - Case A
# ============================================

'''
    NPY_DIR2 = 'datasets/dataset-combustion/npy-PS44'

    INPUT_1 = 'R'
    INPUT_2 = 'G'
    INPUT_3 = 'B'

    OUTPUT = 'ts'

    x1 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_1 + '.npy')))[20:,:,:]
    x2 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_2 + '.npy')))[20:,:,:]
    x3 = np.load(os.path.abspath(os.path.join(NPY_DIR2, INPUT_3 + '.npy')))[20:,:,:]

    x1 = x1[:,:,:32]
    x2 = x2[:,:,:32]
    x3 = x3[:,:,:32]

    for i in range(len(x2)):
        x_max = np.max([x1[i].max(),x2[i].max(),x3[i].max()])
        x2[i] = x2[i][::-1]/x_max
        x3[i] = x3[i][::-1]/x_max
        x1[i] = x1[i][::-1]/x_max

    x1_mean, x1_std = x1.mean(), x1.std()
    x2_mean, x2_std = x2.mean(), x2.std()
    x3_mean, x3_std = x3.mean(), x3.std()    
'''

'''
X1_MEAN  = 0.015027713587521403
X1_STD   = 0.04165276409824668

X2_MEAN  = 0.06956058777233882
X2_STD   = 0.1894216268665299

X3_MEAN  = 0.06211554776822282
X3_STD   = 0.16739457536344302

X_MAX    = 235.41109536012843
INV_XMAX = np.float32(1/X_MAX)     
'''
   
# ============================================
# DATASET NORMALIZATION CONSTANTS - All cases
# ============================================
    
'''
DATASETS = [
    'datasets/dataset-combustion/npy-PS44',
    'datasets/dataset-combustion/npy-PSB40-4',
    'datasets/dataset-combustion/npy-PSB60',
    'datasets/dataset-combustion/npy-PSB80'
]

INPUTS = ['R', 'G', 'B']
all_x1, all_x2, all_x3 = [], [], []
global_max = 0.0

# Cargar y encontrar X_MAX global
for dataset in DATASETS:
    x1 = np.load(os.path.join(dataset, 'R.npy'))[20:, :, :32]
    x2 = np.load(os.path.join(dataset, 'G.npy'))[20:, :, :32]
    x3 = np.load(os.path.join(dataset, 'B.npy'))[20:, :, :32]
    for i in range(len(x1)):
        frame_max = max(x1[i].max(), x2[i].max(), x3[i].max())
        global_max = max(global_max, frame_max)

# Normalizar + flip + acumular
for dataset in DATASETS:
    x1 = np.load(os.path.join(dataset, 'R.npy'))[20:, :, :32]
    x2 = np.load(os.path.join(dataset, 'G.npy'))[20:, :, :32]
    x3 = np.load(os.path.join(dataset, 'B.npy'))[20:, :, :32]
    for i in range(len(x1)):
        all_x1.append(x1[i][::-1] / global_max)
        all_x2.append(x2[i][::-1] / global_max)
        all_x3.append(x3[i][::-1] / global_max)

# Convertir a arrays
all_x1 = np.array(all_x1)
all_x2 = np.array(all_x2)
all_x3 = np.array(all_x3)

# Estadísticas finales
X1_MEAN, X1_STD = all_x1.mean(), all_x1.std()
X2_MEAN, X2_STD = all_x2.mean(), all_x2.std()
X3_MEAN, X3_STD = all_x3.mean(), all_x3.std()
X_MAX = global_max
'''    

X1_MEAN = 0.003930921760659862
X1_STD  = 0.01091675497061941

X2_MEAN = 0.019108146307200424
X2_STD  = 0.051508108170714

X3_MEAN = 0.01762230914738107
X3_STD  = 0.046621915109236536

X_MAX   = 2005.2559796039604
INV_XMAX= np.float32(1/X_MAX)