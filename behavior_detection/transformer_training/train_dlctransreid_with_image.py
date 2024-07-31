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
from behavior_detection.transformer_training.make_model_dlc_and_image import make_dlc_model
from behavior_detection.transformer_training.make_model_dlc_and_image import make_dlc_model_just_image
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


def split_train_test(npy_list, train_frac, npy_shape, dtype='float32', train_map_path='train_data.mmap', test_map_path='test_data.mmap', seed=42, chunk_size=1024):
    if seed is not None:
        np.random.seed(seed)
        
    print(f"Creating train and test maps at {train_map_path}, {test_map_path}")
    
    total_train_samples = 0
    total_test_samples = 0
    
    print('Calculating total samples for train and test datasets')
    for npy in npy_list:
        mmap = np.memmap(npy, dtype=dtype, mode='r', shape=npy_shape)
        n_samples = mmap.shape[0]
        num_train = int(n_samples * train_frac)
        total_train_samples += num_train
        total_test_samples += n_samples - num_train

    train_shape = (total_train_samples,) + npy_shape[1:]
    test_shape = (total_test_samples,) + npy_shape[1:]
    
    print('Creating memmap files')
    train_mmap = np.memmap(train_map_path, dtype=dtype, mode='w+', shape=train_shape)
    test_mmap = np.memmap(test_map_path, dtype=dtype, mode='w+', shape=test_shape)
    
    train_start_idx = 0
    test_start_idx = 0
    
    print('Writing data to memmap files')
    for npy in npy_list:
        mmap = np.memmap(npy, dtype=dtype, mode='r', shape=npy_shape)
        n_samples = mmap.shape[0]
        indices = np.random.permutation(n_samples)
        num_train = int(n_samples * train_frac)
        
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        # Write train data in chunks
        for i in range(0, len(train_indices), chunk_size):
            chunk_indices = train_indices[i:i+chunk_size]
            train_end_idx = train_start_idx + len(chunk_indices)
            train_mmap[train_start_idx:train_end_idx] = mmap[chunk_indices]
            train_start_idx = train_end_idx
            
        # Write test data in chunks
        for i in range(0, len(test_indices), chunk_size):
            chunk_indices = test_indices[i:i+chunk_size]
            test_end_idx = test_start_idx + len(chunk_indices)
            test_mmap[test_start_idx:test_end_idx] = mmap[chunk_indices]
            test_start_idx = test_end_idx

    # Flush to ensure data is written to disk
    train_mmap.flush()
    test_mmap.flush()
    
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
    just_image = False,
    feature_extractor_in_dim=None,
    feature_extractor_out_dim=None,
    npy_list_filenames = None,
    train_map_path = 'train_data.mmap',
    test_map_path = 'test_data.mmap'
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
    
    if just_image:
        model = make_dlc_model_just_image(cfg, 
                                        feature_dim,
                                        num_kpts,
                                        feature_extractor,
                                        feature_extractor_in_dim,
                                        feature_extractor_out_dim)
        print("got dlc model just images")
    else:
        # make my own model factory
        model = make_dlc_model(cfg, 
                            feature_dim,
                            num_kpts,
                            feature_extractor,
                            feature_extractor_in_dim,
                            feature_extractor_out_dim)
        print("got dlc model that includes features")
    
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
