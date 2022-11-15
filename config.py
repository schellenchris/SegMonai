"""!
@file config.py
Sets the parameters for configuration
"""
import socket
import torch
import numpy as np
from enum import Enum
from monai.networks.nets import UNet
from monai.losses import DiceLoss


class NOISE(Enum):
    GAUSSIAN = 0
    POISSON = 1


class SAMPLING(Enum):
    UNIFORM = 0
    CONSTRAINED_MUSTD = 1
    CONSTRAINED_LABEL = 2


class NORMALIZING(Enum):
    WINDOW = 0
    MEAN_STD = 1
    PERCENT5 = 2


class ORGAN(Enum):
    LIVER = 1


def set_attrs(obj, **kwargs):
    for k, v in kwargs.items():
        setattr(obj, k, v)


target_organ = ORGAN.LIVER

train_csv = 'train.csv'
fine_csv = 'fine.csv'
val_csv = 'val.csv'
test_csv = 'test.csv'

prediction_file_name_prefix = 'prediction-'
sample_file_name_prefix = 'ct-volume-'  # 'mr-volume-'
label_file_name_prefix = 'liver-segmentation-'
ensemble_name = 'Combination'

# Mode
verbose = True

if socket.gethostname() == 'ckm4cad':
    on_server = True
    op_parallelism_threads = 6
    batch_size_train = 4
    batch_capacity_train = 4000
    train_reader_instances = 2
else:
    on_server = False
    op_parallelism_threads = 3
    batch_size_train = 4
    batch_capacity_train = 50
    train_reader_instances = 1

training_epochs = 200
min_n_epochs = 30
checkpoints_to_keep = 10
validation_values_to_keep = 10

do_gradient_clipping = False
clipping_value = 50

# Testing
do_connected_component_analysis = False
do_filter_small_components = False
min_number_of_voxels = 15
batch_size_test = 1
summaries_per_case = 10
write_probabilities = False

# Data
num_channels = 1
num_slices = 1
num_slices_train = 32
num_slices_test = 32
num_classes_seg = 1
num_files = -1
train_dim = 512
train_input_shape = [num_channels, train_dim, train_dim]
train_label_shape = [num_classes_seg, train_dim, train_dim]
test_dim = 512
test_data_shape = [num_channels, test_dim, test_dim]
test_label_shape = [num_classes_seg, test_dim, test_dim]

dtype = torch.float32
data_train_split = 0.75
number_of_val = 2

# Loader
val_reader_instances = 1
file_name_capacity = 140
file_name_capacity_valid = file_name_capacity // 10
batch_capacity_valid = batch_capacity_train // 2
normalizing_method = NORMALIZING.WINDOW    # NORMALIZING.MEAN_STD

# Sample Mining
patch_shift_factor = 3  # 3*std is 99th percentile
in_between_slice_factor = 2
min_n_samples = 10
random_sampling_mode = SAMPLING.UNIFORM  # SAMPLING_MODES.CONSTRAINED_LABEL
percent_of_object_samples = 50  # %
samples_per_volume = 40
samples_per_slice_object = 2
samples_per_slice_lesion = 4
samples_per_slice_bkg = 1
samples_per_slice_uni = 1
do_flip_coronal = False
do_flip_sagittal = False
do_variate_intensities = False
intensity_variation_interval = 0.01
do_deform = False
deform_sigma = 10  # standard deviation of the normal distribution
points = 3  # size of the grid (3x3 grid)
add_noise = False
noise_typ = NOISE.POISSON
standard_deviation = 30
mean_poisson = 70

# Resampling
adapt_resolution = True
if adapt_resolution:
    target_spacing = [0.75, 0.75, 1.5]
    target_size = [512, 512]
target_direction = 'PLI'  # make sure all images are oriented equally
target_type_image = np.float32
target_type_label = np.uintc
data_background_value = 0  # Dominik, CT:-1000
label_background_value = 0
max_rotation = 0  # 0.07*pi =12,6°  0.05*pi = 9°

# Tversky
tversky_alpha = 0.3
tversky_beta = 1 - tversky_alpha

# Weighted CE
basis_factor = 5
tissue_factor = 5
contour_factor = 2
max_weight = 1.2
tissue_threshold = -0.9

# Preprocessing
norm_min_v = 0  # -200 #-150
norm_max_v = 200  # 2660 #250    # 275
norm_eps = 1e-5

# Model
architecture = UNet

# Training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr = 1e-3
optim = torch.optim.Adam
loss = DiceLoss()
