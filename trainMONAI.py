import os
import glob
import numpy as np
import pandas as pd
import config as cfg
from monai.networks.nets import UNet

from monai.data import ArrayDataset, create_test_image_3d, decollate_batch, DataLoader, ImageDataset
from monai.utils import set_determinism
from monai.data.utils import list_data_collate, decollate_batch, no_collation
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandSpatialCrop,
    CenterSpatialCrop,
    Resized,
    ScaleIntensity,
    ResizeWithPadOrCrop,
    Orientation,
    Spacing,
    RandRotate,
    RandFlip,
    RandGaussianNoise,
    RandShiftIntensity,
    NormalizeIntensity,
    RandSpatialCropSamples,
    CropForeground
)
import ignite
import torch
from monai.losses import DiceLoss, DiceCELoss
from monai.handlers import StatsHandler
from monai.utils import first
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events

experiment_name = "liver_seg"
logs_path = os.path.join("tmp", experiment_name)


def _make_folder_name(hyper_parameters, seed):
    epochs = cfg.training_epochs // 10
    windowing = 'Win' + str(cfg.norm_max_v)
    patches = 'Dim' + str(cfg.train_dim)

    if hyper_parameters['init_parameters']['drop_out'][0]:
        do = 'DO'
    else:
        do = 'nDO'

    if hyper_parameters['init_parameters']['do_batch_normalization']:
        bn = 'BN'
    else:
        bn = 'nBN'
    augment = ""
    if cfg.do_flip_coronal or cfg.do_flip_sagittal or cfg.do_variate_intensities or cfg.do_deform or cfg.add_noise or cfg.max_rotation != 0:
        if cfg.do_flip_coronal:
            augment += 'FCor'
        if cfg.do_flip_sagittal:
            augment += 'FSag'
        if cfg.do_variate_intensities:
            augment += 'VI' + str(cfg.intensity_variation_interval)
        if cfg.do_deform:
            augment += 'Def'
        if cfg.add_noise:
            if cfg.noise_typ == cfg.noise_typ.POISSON:
                augment += 'PoiNoi'
            elif cfg.noise_typ == cfg.noise_typ.GAUSSIAN:
                augment += 'GauNoi'
        if cfg.max_rotation != 0:
            augment += 'Rot' + str(cfg.max_rotation)

    else:
        augment = 'nAug'
    folder_name = "-".join([hyper_parameters['architecture'].get_name() + str(hyper_parameters['dimensions']) + 'D',
                            hyper_parameters['loss'], do, bn, augment, windowing, patches, str(seed)])

    return folder_name


def _set_parameters_according_to_dimension(hyper_parameters):
    cfg.set_attrs(
        cfg,
        num_channels=1,
        train_dim=512,
        samples_per_volume=20,
        batch_capacity_train=40,  # 250
        batch_capacity_valid=20,  # 150
        num_slices_train=16,
        train_input_shape=[cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_channels],
        train_label_shape=[cfg.num_slices_train, cfg.train_dim, cfg.train_dim, cfg.num_classes_seg],
        test_dim=512,
        num_slices_test=32,
        test_data_shape=[cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_channels],
        test_label_shape=[cfg.num_slices_test, cfg.test_dim, cfg.test_dim, cfg.num_classes_seg],
        batch_size_train=2,
        batch_size_test=1,
        architecture=UNet,
        lr=1e-3,
        optim=torch.optim.Adam,  # SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        loss=DiceLoss()  # DiceCELoss()
    )
        
    print('   Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
    print('   Test Shapes: ', cfg.test_data_shape, cfg.test_label_shape)


def get_transpose():
    im_trans = Compose(
        [
            EnsureChannelFirst(),
            Spacing(pixdim=cfg.target_spacing),
            Orientation(axcodes='PLI'),
            NormalizeIntensity(),
            ResizeWithPadOrCrop([512, 512, 64], mode='minimum'),
            RandRotate(range_z=[12.6, 12.6], prob=0.25),
            # CenterSpatialCrop((512, 512, 64)),
            # CropForeground(),
            # RandSpatialCrop((256, 256, 32), random_size=False),
            # RandFlip(prob=0.5, spatial_axis=1),
            # RandGaussianNoise(mean=0, std=0.01),
            # RandShiftIntensity(prob=0.5, offsets=(10, 20))
        ]
    )

    seg_trans = Compose(
        [
            EnsureChannelFirst(),
            Spacing(pixdim=cfg.target_spacing),
            Orientation(axcodes='PLI'),
            ResizeWithPadOrCrop([512, 512, 64], mode='minimum'),
            RandRotate(range_z=[12.6, 12.6], prob=0.25),
            # CenterSpatialCrop((512, 512, 64)),
            # CropForeground(),
            # RandSpatialCrop((256, 256, 32), random_size=False),
            # RandFlip(prob=1, spatial_axis=1)
            # AsDiscrete(to_onehot=2)
        ]
    )
    return im_trans, seg_trans


def get_filenames(file=cfg.train_csv):
    files = pd.read_csv(file, dtype=object).values
    files = [(os.path.split(file_id[0])[0], os.path.split(file_id[0])[1]) for file_id in files]
    imgs = [os.path.join(folder, cfg.sample_file_name_prefix + number + '.nii') for folder, number in files]
    segs = [os.path.join(folder, cfg.label_file_name_prefix + number + '.nii') for folder, number in files]
    return imgs, segs


def get_loader(file=cfg.train_csv):
    img_trans, seg_trans = get_transpose()
    imgs, segs = get_filenames(file)
    ds = ImageDataset(
        image_files=imgs,
        seg_files=segs,
        transform=img_trans,
        seg_transform=seg_trans,
        image_only=True,
    )
    return DataLoader(
        ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available(), shuffle=True
    )


def get_network():
    return cfg.architecture(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(8, 32, 64, 128),  # 128), #256),
        strides=(2, 2, 2),
        num_res_units=3,
        dropout=0.05
    ).to(cfg.device)


def save_checkpoints(net, optim, trainer, path, net_name='net'):
    checkpoint_handler = ModelCheckpoint(path, net_name, n_saved=10, require_empty=False)
    trainer.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=checkpoint_handler, to_save={"net": net, "opt": optim}
    )
    torch.save(net, f'{path}/{net_name}.pt')
    torch.save(optim, f'{path}/{net_name}_optim.pt')


def load_checkpoint(path, net_name='net', checkpoint=None):
    net = torch.load(f'{path}/{net_name}.pt')
    optim = torch.load(f'{path}/{net_name}_optim.pt')
    objects_to_checkpoint = {"net": net, "opt": optim}
    list_of_files = glob.glob(f'{path}/*.pt')
    latest_file = max([file for file in list_of_files if 'checkpoint' in file], key=os.path.getctime)
    ModelCheckpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=torch.load(latest_file))
    return net, optim


def train(seed=42):
    np.random.seed(seed)
    set_determinism(seed=seed)
    loader = get_loader()
    net = get_network()
    optim = cfg.optim(net.parameters(), cfg.lr)
    trainer = ignite.engine.create_supervised_trainer(
        net, optim, cfg.loss, cfg.device
    )
    save_checkpoints(net, optim, trainer, "./runs_array/DiceCELoss/")
    StatsHandler(name='trainer', output_transform=lambda x: x).attach(trainer)
    state = trainer.run(loader, cfg.training_epochs)


def experiment_3(data, hyper_parameters, k_fold, dimensions_and_architectures, losses, augment_val):
    # Experiment 3: Test best data augmentation combinations
    for (data_name, data_set) in data:
        np.random.seed(42)

        hyper_parameters['experiment_path'] = os.path.join(logs_path, 'experiment3-Server-' + data_name)
        all_indices = np.random.permutation(range(0, data_set.size))
        test_folds = np.array_split(all_indices, k_fold)

        for f in range(k_fold):
            test_indices = test_folds[f]
            remaining_indices = np.setdiff1d(all_indices, test_indices)
            val_indices = remaining_indices[:cfg.number_of_val]
            train_indices = remaining_indices[cfg.number_of_val:]

            train_files = data_set[train_indices]
            val_files = data_set[val_indices]
            test_files = data_set[test_indices]
            #train_files = np.vstack([train_files, test_files])

            np.savetxt(cfg.train_csv, train_files, fmt='%s', header='path')
            np.savetxt(cfg.val_csv, val_files, fmt='%s', header='path')
            np.savetxt(cfg.test_csv, test_files, fmt='%s', header='path')

            cfg.num_files = len(train_files)

            print('  Data Set ' + data_name + str(f) + ': ' + str(train_indices.size) + ' train cases, '
                  + str(test_indices.size)
                  + ' test cases, ' + str(val_indices.size) + ' val cases')

            for d, a in dimensions_and_architectures:
                hyper_parameters["dimensions"] = d
                _set_parameters_according_to_dimension(hyper_parameters)
                hyper_parameters['architecture'] = a
                for l in losses:
                    hyper_parameters["loss"] = l
                    for cfg.do_flip_sagittal, cfg.max_rotation, cfg.add_noise, cfg.do_variate_intensities in augment_val:

                        try:
                            cfg.random_sampling_mode = cfg.SAMPLING.UNIFORM
                            cfg.percent_of_object_samples = 50
                            train(seed=f)
                            pass
                        except Exception as err:
                            print('Training ' + data_name,
                                  'UNET' + hyper_parameters['loss'] + 'failed!')
                            print(err)


if __name__ == '__main__':
    np.random.seed(42)
    chaos_csv = 'chaos.csv'
    all_chaos_files = pd.read_csv(chaos_csv, dtype=object).values
    k_fold = 2
    losses = ['CEL+DICE']
    dimensions_and_architectures = ([3, UNet])

    init_parameters = {"regularize": [True, 'L2', 0.0000001], "drop_out": [True, 0.01], "activation": "elu",
                       "do_batch_normalization": False, "do_bias": True, "cross_hair": False}
    train_parameters = {"l_r": 0.001, "optimizer": "Adam", "early_stopping": [True, cfg.min_n_epochs, 1e-4]}
    hyper_parameters = {"init_parameters": init_parameters, "train_parameters": train_parameters}

    windowing = [(-200, 200), (-200, 250), (-100, 400)]
    train_dim = [256, 512]

    # load_checkpoint("./runs_array/DiceCELoss/")
    train()


