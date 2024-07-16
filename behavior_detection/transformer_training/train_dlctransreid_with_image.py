#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import random

try:
    import torch
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Unsupervised identity learning requires PyTorch. Please run `pip install torch`."
    )
import numpy as np
import os
import glob
from pathlib import Path
from deeplabcut.pose_tracking_pytorch.config import cfg
from deeplabcut.pose_tracking_pytorch.datasets import make_dlc_dataloader
from behavior_detection.transformer_training.make_model_image_input import make_dlc_model
from deeplabcut.pose_tracking_pytorch.solver import make_easy_optimizer
from deeplabcut.pose_tracking_pytorch.solver.scheduler_factory import create_scheduler
from deeplabcut.pose_tracking_pytorch.loss import easy_triplet_loss
from deeplabcut.pose_tracking_pytorch.processor import do_dlc_train


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def split_train_test(npy_list, train_frac, npy_shape, dtype='float32', train_map_path='train_data.mmap', test_map_path='test_data.mmap', seed=42, overwrite=False):
    if seed is not None:
        np.random.seed(seed)
        
    train_list = []
    test_list = []
    print(f"creating train and test maps at {train_map_path}, {test_map_path}")
    if not os.path.isfile(train_map_path) or not os.path.isfile(test_map_path) or overwrite:
        # if file exists we create, otherwise we just return the file
        print('getting indices')
        for npy in npy_list:
            # using memmap here because it can't fit in RAM, memmap leaves on disk
            mmap = np.memmap(npy, dtype=dtype, mode='r', shape=npy_shape)
            n_samples = mmap.shape[0]
            indices = np.random.permutation(n_samples)
            num_train = int(n_samples * train_frac)
            
            train_indices = indices[:num_train]
            test_indices = indices[num_train:]
            
            train = mmap[train_indices]
            test = mmap[test_indices]
            
            train_list.append(train)
            test_list.append(test)
        print('getting num samples')
        # Create memory-mapped files for the concatenated train and test data
        total_train_samples = sum(len(train) for train in train_list)
        total_test_samples = sum(len(test) for test in test_list)

        train_shape = (total_train_samples,) + npy_shape[1:]
        test_shape = (total_test_samples,) + npy_shape[1:]
        print('creating memmap')
        train_mmap = np.memmap(train_map_path, dtype=dtype, mode='w+', shape=train_shape)
        test_mmap = np.memmap(test_map_path, dtype=dtype, mode='w+', shape=test_shape)
    else:
        train_shape = (total_train_samples,) + npy_shape[1:]
        test_shape = (total_test_samples,) + npy_shape[1:]
        
        train_mmap = np.memmap(train_map_path, dtype=dtype, mode='r', shape=train_shape)
        test_mmap = np.memmap(test_map_path, dtype=dtype, mode='r', shape=test_shape)
        return train_mmap, test_mmap
    
    # Copy data to the memory-mapped files
    start_idx = 0
    for train in train_list:
        end_idx = start_idx + len(train)
        train_mmap[start_idx:end_idx] = train
        start_idx = end_idx

    start_idx = 0
    for test in test_list:
        end_idx = start_idx + len(test)
        test_mmap[start_idx:end_idx] = test
        start_idx = end_idx

    return train_mmap, test_mmap


def train_tracking_transformer_image(
    path_config_file,
    dlcscorer,
    videos,
    videotype="",
    train_frac=0.8,
    modelprefix="",
    n_triplets=1000,
    train_epochs=100,
    batch_size=64,
    ckpt_folder="",
    destfolder=None,
    feature_dim=2048,
    num_kpts=9,
    feature_extractor=None,
    feature_extractor_in_dim=None,
    feature_extractor_out_dim=None,
    npy_list_filenames = None,
    train_map_path = 'train_data.mmap',
    test_map_path = 'test_data.mmap',
):
    npy_list = []
    if npy_list_filenames is None:
        from deeplabcut.utils import auxiliaryfunctions
        videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
        for video in videos:
            videofolder = str(Path(video).parents[0])
            if destfolder is None:
                destfolder = videofolder
            video_name = Path(video).stem
            # video_name = '.'.join(video.split("/")[-1].split(".")[:-1])
            files = glob.glob(os.path.join(destfolder, video_name + dlcscorer + "*image.npy"))

            # assuming there is only one match
            npy_list.append(files[0])
    else:
        if type(npy_list_filenames) == list:
            for f in npy_list_filenames:
                npy_list.append(f)
        else:
            # assuming just a single filename
            npy_list.append(npy_list_filenames)

    print('getting lists')
    npy_shape = (n_triplets, 3, num_kpts*feature_dim + np.prod(feature_extractor_in_dim))
    train_list, test_list = split_train_test(npy_list, train_frac, npy_shape, train_map_path=train_map_path, test_map_path=test_map_path)
    print("got lists")

    train_loader, val_loader = make_dlc_dataloader(train_list, test_list, batch_size)
    print("got dataloaders")
    
    # make my own model factory
    model = make_dlc_model(cfg, 
                           feature_dim,
                           num_kpts,
                           feature_extractor,
                           feature_extractor_in_dim,
                           feature_extractor_out_dim)
    print("got model")
    
    # make my own loss factory
    triplet_loss = easy_triplet_loss()

    optimizer = make_easy_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    print('got optimizer and scheduler')
    
    num_query = 1

    do_dlc_train(
        cfg,
        model,
        triplet_loss,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_kpts,
        feature_dim,
        num_query,
        total_epochs=train_epochs,
        ckpt_folder=ckpt_folder,
    )
