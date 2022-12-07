import os
import glob
import warnings

import numpy as np
import pandas as pd
import config as cfg
from monai.networks.nets import UNet

from monai.data import (ArrayDataset, create_test_image_3d, decollate_batch,
                        DataLoader, ImageDataset, PatchIterd, Dataset, GridPatchDataset, CacheDataset)
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
    CropForeground,
    EnsureTyped,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    NormalizeIntensityd,
    RandRotated,
    EnsureTyped
)
import ignite
import torch
from monai.losses import DiceLoss, DiceCELoss
from monai.handlers import StatsHandler, MeanDice, TensorBoardStatsHandler
from monai.utils import first
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events, _prepare_batch
import nibabel as nib

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


def _set_config():
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
    print('Train Shapes: ', cfg.train_input_shape, cfg.train_label_shape)
    print('Test Shapes: ', cfg.test_data_shape, cfg.test_label_shape)


def get_transform(dict_transform=False):
    if dict_transform:
        return Compose(
            [
                LoadImaged(keys=["img", "seg"]),
                EnsureChannelFirstd(keys=["img", "seg"]),
                # Spacingd(keys=["img", "seg"], pixdim=cfg.target_spacing),
                Orientationd(keys=["img", "seg"], axcodes='PLI'),
                NormalizeIntensityd(keys='img'),
                RandRotated(keys=["img"], range_z=(12.6, 12.6), prob=0.25),
                EnsureTyped(keys=["img", "seg"]),
            ]
        )
    img_trans = Compose(
        [
            EnsureChannelFirst(),
            # Spacing(pixdim=cfg.target_spacing),
            Orientation(axcodes='PLI'),
            NormalizeIntensity(),
            RandRotate(range_z=(12.6, 12.6), prob=0.25),
            ResizeWithPadOrCrop([512, 512, 64], mode='minimum')
        ]
    )
    seg_trans = Compose(
        [
            EnsureChannelFirst(),
            # Spacing(pixdim=cfg.target_spacing),
            Orientation(axcodes='PLI'),
            ResizeWithPadOrCrop([512, 512, 64], mode='minimum')
        ]
    )
    return img_trans, seg_trans


def get_filenames(file=None, k_fold=1, val_size=None):
    file = cfg.train_csv if file is None else file
    files = pd.read_csv(file, dtype=object).values
    files = [(os.path.split(file_id[0])[0], os.path.split(file_id[0])[1]) for file_id in files]
    img_files = np.array([os.path.join(folder, cfg.sample_file_name_prefix + number + '.nii') for folder, number in files])
    seg_files = np.array([os.path.join(folder, cfg.label_file_name_prefix + number + '.nii') for folder, number in files])
    all_indices = np.random.permutation(range(len(img_files)))
    test_folds = np.array_split(all_indices, k_fold)
    k_folds = []
    for k in range(k_fold):
        test_indices = test_folds[k]
        remaining_indices = np.setdiff1d(all_indices, test_indices)
        val_indices = remaining_indices[:val_size] if val_size else remaining_indices[:cfg.number_of_val]
        train_indices = remaining_indices[val_size:] if val_size else remaining_indices[cfg.number_of_val:]
        test_img_files, test_seg_files = img_files[test_indices], seg_files[test_indices]
        train_img_files, train_seg_files = img_files[train_indices], seg_files[train_indices]
        val_img_files, val_seg_files = img_files[val_indices], seg_files[val_indices]
        k_folds.append(
            {
                'train': {
                    'img_files': train_img_files,
                    'seg_files': train_seg_files
                },
                'val': {
                    'img_files': val_img_files,
                    'seg_files': val_seg_files
                },
                'test': {
                    'img_files': test_img_files,
                    'seg_files': test_seg_files
                }
            }
        )
    return k_folds


def get_loader(img_files=None, seg_files=None, file=None):
    if img_files is None:
        file = cfg.train_csv if file is None else file
        img_files, seg_files = get_filenames(file)
    img_trans, seg_trans = get_transform()
    ds = ImageDataset(
        image_files=img_files,
        seg_files=seg_files,
        transform=img_trans,
        seg_transform=seg_trans,
        image_only=True,
    )
    return DataLoader(
        ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available(), shuffle=False
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


def load_checkpoint(path=None, net_name='net', checkpoint=None):
    path = './checkpoints/' if path is None else path
    net = torch.load(f'{path}/{net_name}.pt')
    optim = torch.load(f'{path}/{net_name}_optim.pt')
    objects_to_checkpoint = {"net": net, "opt": optim}
    list_of_files = glob.glob(f'{path}/*.pt')
    if checkpoint is None:
        checkpoint = max([file for file in list_of_files if 'checkpoint' in file], key=os.path.getctime)
        warnings.warn(f'No checkpoint specified loading {checkpoint}')
    ModelCheckpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=torch.load(checkpoint))
    return net, optim


def train(network=None, data_loader=None, val_loader=None, validation_after=None, dict_transforms=False, epochs=None,
          optimizer=None, loss=None, checkpoint_folder=None, seed=42):
    np.random.seed(seed)
    set_determinism(seed=seed)
    net = get_network() if network is None else network
    loader = get_loader() if data_loader is None else data_loader
    optim = cfg.optim(net.parameters(), cfg.lr) if optimizer is None else optimizer
    trainer = ignite.engine.create_supervised_trainer(
        net, optim, cfg.loss if loss is None else loss, cfg.device,
        prepare_batch=lambda batch, device, non_blocking:
        _prepare_batch((batch["img"], batch["seg"]), device, non_blocking) if dict_transforms else _prepare_batch
    )
    if val_loader is not None:
        val_metrics = {'Mean_Dice': MeanDice()}
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = Compose([AsDiscrete(threshold=0.5)])
        evaluator = ignite.engine.create_supervised_evaluator(
            net,
            val_metrics,
            cfg.device,
            True,
            output_transform=lambda x, y, y_pred: ([post_pred(i) for i in decollate_batch(y_pred)],
                                                   [post_label(i) for i in decollate_batch(y)]),
            prepare_batch=lambda batch, device, non_blocking:
            _prepare_batch((batch["img"], batch["seg"]), device, non_blocking) if dict_transforms else _prepare_batch
        )

        @trainer.on(Events.ITERATION_COMPLETED(every=validation_after))
        def run_validation(engine):
            evaluator.run(val_loader)
    if checkpoint_folder is None:
        checkpoint_folder = './checkpoints/'
        warnings.warn(f'No checkpoint folder specified, using {checkpoint_folder}')
    save_checkpoints(net, optim, trainer, checkpoint_folder)
    StatsHandler(name='trainer', output_transform=lambda x: x).attach(trainer)
    TensorBoardStatsHandler(output_transform=lambda x: x).attach(trainer)
    state = trainer.run(loader, cfg.training_epochs if epochs is None else epochs)
    return state


def train_k_fold(network=None, k_fold=1, dict_transforms=False, epochs=None, optimizer=None, loss=None, checkpoint_folder=None, seed=42):
    files = get_filenames(k_fold=k_fold)
    for fold in files:
        train_loader = get_loader(fold['train']['img_files'], fold['train']['seg_files'])
        val_loader = get_loader(fold['val']['img_files'], fold['val']['seg_files'])
        test_loader = get_loader(fold['test']['img_files'], fold['test']['seg_files'])
        train(data_loader=train_loader, val_loader=val_loader)


def set_device(device_id=0, name=None):
    if not torch.cuda.is_available():
        warnings.warn('No available CUDA device found, setting device to CPU')
        cfg.device = torch.device('cpu')
    if name is not None:
        devices = {i: torch.cuda.get_device_properties(i).name.lower() for i in range(torch.cuda.device_count())}
        count = 0
        for key in devices.keys():
            if name.lower() in devices[key]:
                count += 1
                device_id = key
        if count > 1:
            warnings.warn(f'Found multiple GPUs matching the name {name}')
    print(f'Setting device to {torch.cuda.get_device_properties(device_id).name}')
    cfg.device = torch.device(f'cuda:{device_id}')


def get_slice_loader(img_files=None, seg_files=None, file=None, cache=True):
    if img_files is None:
        file = cfg.train_csv if file is None else file
        img_files, seg_files = get_filenames(file)
    train_files = [{"img": img, "seg": seg} for img, seg in zip(img_files, seg_files)]
    patch_func = PatchIterd(
        keys=["img", "seg"],
        patch_size=(None, None, 32),  # dynamic first two dimensions
        start_pos=(0, 0, 0),
        mode='minimum'
    )
    patch_transform = Compose(
        [
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    train_transforms = get_transform(dict_transform=True)
    volume_ds = Dataset(data=train_files, transform=train_transforms)
    shapes = [nib.load(img).shape for img in img_files] if cache else None
    patch_ds = GridPatchDataset(
        data=volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False, shapes=shapes)
    if cache:
        patch_ds = CacheDataset(data=patch_ds, transform=None)
    return DataLoader(patch_ds, batch_size=1, num_workers=2, pin_memory=torch.cuda.is_available(), shuffle=False)


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
            # train_files = np.vstack([train_files, test_files])

            np.savetxt(cfg.train_csv, train_files, fmt='%s', header='path')
            np.savetxt(cfg.val_csv, val_files, fmt='%s', header='path')
            np.savetxt(cfg.test_csv, test_files, fmt='%s', header='path')

            cfg.num_files = len(train_files)

            print('  Data Set ' + data_name + str(f) + ': ' + str(train_indices.size) + ' train cases, '
                  + str(test_indices.size)
                  + ' test cases, ' + str(val_indices.size) + ' val cases')

            for d, a in dimensions_and_architectures:
                hyper_parameters["dimensions"] = d
                # _set_parameters_according_to_dimension(hyper_parameters)
                _set_config()
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
    # np.random.seed(42)
    chaos_csv = 'chaos.csv'
    # all_chaos_files = pd.read_csv(chaos_csv, dtype=object).values
    # k_fold = 2 TODO
    # losses = ['CEL+DICE']
    # dimensions_and_architectures = ([3, UNet])
    #
    # init_parameters = {"regularize": [True, 'L2', 0.0000001], "drop_out": [True, 0.01], "activation": "elu",
    #                    "do_batch_normalization": False, "do_bias": True, "cross_hair": False}
    # train_parameters = {"l_r": 0.001, "optimizer": "Adam", "early_stopping": [True, cfg.min_n_epochs, 1e-4]}
    # hyper_parameters = {"init_parameters": init_parameters, "train_parameters": train_parameters}
    #
    # windowing = [(-200, 200), (-200, 250), (-100, 400)]
    # train_dim = [256, 512]
    #
    # net, optim = load_checkpoint()
    # loader = get_slice_loader()
    # check_data = first(loader)
    # print("first volume's shape: ", check_data["img"].shape, check_data["seg"].shape)
    # train(data_loader=get_slice_loader(), dict_transforms=True)
    # set_device(name='gtx 1050')
    train_k_fold(k_fold=3)
