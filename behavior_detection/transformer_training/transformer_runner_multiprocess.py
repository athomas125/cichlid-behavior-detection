import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from deeplabcut.utils import auxiliaryfunctions
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from deeplabcut.pose_tracking_pytorch.tracking_utils.meter import AverageMeter
from deeplabcut.pose_tracking_pytorch.tracking_utils.metrics import R1_mAP_eval
from deeplabcut.pose_tracking_pytorch.datasets.dlc_vec import TripletDataset
from deeplabcut.pose_tracking_pytorch.model import make_dlc_model
from deeplabcut.pose_tracking_pytorch.loss import easy_triplet_loss
from deeplabcut.pose_tracking_pytorch.solver import make_easy_optimizer
from deeplabcut.pose_tracking_pytorch.solver.scheduler_factory import create_scheduler
from deeplabcut.pose_tracking_pytorch.config import cfg

from pathlib import Path
import numpy as np
import glob
import logging
import time
import pickle


def distance_func(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1))

def easy_triplet_loss(anchor, positive, negative):
    triplet_loss = torch.nn.TripletMarginLoss()
    loss = triplet_loss(anchor, positive, negative)
    return loss

def calc_correct(anchor, pos, neg):
    # cos = torch.cdist
    ap_dist = distance_func(anchor, pos)
    an_dist = distance_func(anchor, neg)
    indices = ap_dist < an_dist

    return torch.sum(indices)


def calc_cos_correct(vec1, gt1, vec2, gt2, threshold=0.5):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    confidence = cos(vec1, vec2)

    pred_mask = confidence > threshold

    gt_mask = gt1 == gt2
    n_correct = torch.sum(torch.eq(pred_mask, gt_mask))
    return n_correct


def init_process(rank, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


# modified from dlc.pose_tracking_pytorch.datasets.make_dataloader    
def make_dist_dlc_dataloader(train_list, test_list, batch_size=64):
    train_dataset = TripletDataset(train_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = TripletDataset(test_list)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def split_train_test(npy_list, train_frac):
    # with npy list form videos, split each to train and test

    x_list = []
    train_list = []
    test_list = []

    for npy in npy_list:
        vectors = np.load(npy)
        n_samples = vectors.shape[0]
        indices = np.random.permutation(n_samples)
        num_train = int(n_samples * train_frac)
        vectors = vectors[indices]
        train = vectors[:num_train]
        test = vectors[num_train:]
        train_list.append(train)
        test_list.append(test)

    train_list = np.concatenate(train_list, axis=0)
    test_list = np.concatenate(test_list, axis=0)

    return train_list, test_list


def do_dist_dlc_train(
    rank,
    world_size,
    cfg,
    model,
    triplet_loss,
    train_data_list,# this is a dataloader, this needs to change
    val_data_list, # this is a dataloader, this needs to change
    optimizer, # will this optimizer work in distributed?
    scheduler,
    num_kpts,
    feature_dim,
    num_query,
    total_epochs=100,
    ckpt_folder="",
):
    init_process(rank, world_size)
    
    log_period = cfg["log_period"]
    checkpoint_period = cfg["checkpoint_period"]
    eval_period = 10

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    logger = logging.getLogger("transreid.train")
    if rank == 0:
        logger.info("start training")
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg["feat_norm"])

    # Train and validation loaders with DistributedSampler
    train_dataset = TripletDataset(train_data_list)
    val_dataset = TripletDataset(val_data_list)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], sampler=val_sampler
    )

    # Training
    epoch_list = []
    train_acc_list = []
    test_acc_list = []
    plot_dict = {}
    for epoch in range(1, total_epochs + 1):
        epoch_list.append(epoch)
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        total_n = 0.0
        total_correct = 0.0
        for n_iter, (anchor, pos, neg) in enumerate(train_loader):
            optimizer.zero_grad()

            anchor = anchor.to(device)
            pos = pos.to(device)
            neg = neg.to(device)

            anchor_feat = model(anchor)
            pos_feat = model(pos)
            neg_feat = model(neg)

            loss = triplet_loss(anchor_feat, pos_feat, neg_feat)
            loss.backward()
            optimizer.step()

            total_n += anchor_feat.shape[0]
            total_correct += calc_correct(anchor_feat, pos_feat, neg_feat)

            loss_meter.update(loss.item())

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if (n_iter + 1) % log_period == 0 and rank == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, , Base Lr: {:.2e}".format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        train_acc = total_correct / total_n
        train_acc_list.append(train_acc.item())

        if cfg["dist_train"] and rank == 0:
            pass
        else:
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        model_name = f"dlc_transreid"

        if epoch % checkpoint_period == 0 and rank == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "num_kpts": num_kpts,
                    "feature_dim": feature_dim,
                },
                os.path.join(ckpt_folder, model_name + "_{}.pth".format(epoch)),
            )

        if epoch % eval_period == 0:
            model.eval()
            val_loss = 0.0
            total_n = 0.0
            total_correct = 0.0
            for n_iter, (anchor, pos, neg) in enumerate(val_loader):
                with torch.no_grad():
                    anchor = anchor.to(device)
                    pos = pos.to(device)
                    neg = neg.to(device)
                    anchor_feat = model(anchor)
                    pos_feat = model(pos)
                    neg_feat = model(neg)
                    loss = triplet_loss(anchor_feat, pos_feat, neg_feat)
                    val_loss += loss.item()

                    total_n += anchor_feat.shape[0]
                    total_correct += calc_correct(anchor_feat, pos_feat, neg_feat)

            if rank == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                test_acc = total_correct / total_n
                test_acc_list.append(test_acc.item())
                print(f"Epoch {epoch}, train acc: {train_acc:.2f}")
                print(f"Epoch {epoch}, test acc {test_acc:.2f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if rank == 0:
        plot_dict["train_acc"] = train_acc_list
        plot_dict["test_acc"] = test_acc_list
        plot_dict["epochs"] = epoch_list

        with open(
            os.path.join(ckpt_folder, "dlc_transreid_results.pickle"), "wb"
        ) as handle:
            pickle.dump(plot_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    dist.destroy_process_group()

def main():
    world_size = 8
    trainingsetindex = 0
    epochs = 100

    cfg["batch_size"] =  64
    cfg["dist_train"] =  True
    
    npy_list = []
    # hardcoding file here for now
    npy_list.append('/data/home/athomas314/test_video/MC_singlenuc23_1_Tk33_0212200003_vid_clip_36170_38240DLC_dlcrnetms5_dlc_modelJul26shuffle1_100000_triplet_vector.npy')
    
    # Initialize your model, triplet_loss, train_dataset, val_dataset, optimizer, scheduler, num_kpts, feature_dim, num_query here
    train_list, val_list = split_train_test(npy_list, .95)
    num_kpts = train_list.shape[2]
    feature_dim = train_list.shape[-1]
    model = make_dlc_model(cfg, feature_dim, num_kpts)
    # triplet_loss = easy_triplet_loss()
    optimizer = make_easy_optimizer(cfg, model)
    scheduler = create_scheduler(cfg, optimizer)
    num_query = 1
    ckpt_folder = os.path.abspath('/data/home/athomas314/dlc_model-student-2023-07-26/dlc-models/iteration-3/dlc_modelJul26-trainset95shuffle1/mp_train')

    mp.spawn(do_dist_dlc_train,
             args=(world_size, cfg, model, easy_triplet_loss, train_list, val_list, optimizer, scheduler, num_kpts, feature_dim, num_query, epochs, ckpt_folder),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()
