import sys
import os
from pathlib import Path

# Get the directory of the current file
current_file_dir = Path(__file__).resolve().parent

# Construct the path to the cichlid-behavior-detection directory
project_root = current_file_dir.parent.parent

# Append the project root to sys.path
sys.path.append(str(project_root))

from behavior_detection.transformer_training.ViTFeatureExtractor import ViTFeatureExtractor
from behavior_detection.transformer_training.train_dlctransreid_with_image import train_tracking_transformer_image
from deeplabcut.pose_tracking_pytorch.config import cfg
from deeplabcut.utils import auxiliaryfunctions
import torch
from torchvision.models import vit_h_14, ViT_H_14_Weights
from PIL import Image


    
def return_train_network_path(config, shuffle=1, trainingsetindex=0, modelprefix=""):
    """Returns the training and test pose config file names as well as the folder where the snapshot is
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    shuffle: int
        Integer value specifying the shuffle index to select for training.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    Returns the triple: trainposeconfigfile, testposeconfigfile, snapshotfolder
    """

    cfg = auxiliaryfunctions.read_config(config)
    modelfoldername = auxiliaryfunctions.get_model_folder(
        cfg["TrainingFraction"][trainingsetindex], shuffle, cfg, modelprefix=modelprefix
    )
    trainposeconfigfile = Path(
        os.path.join(
            cfg["project_path"], str(modelfoldername), "train", "pose_cfg.yaml"
        )
    )
    testposeconfigfile = Path(
        os.path.join(cfg["project_path"], str(modelfoldername), "test", "pose_cfg.yaml")
    )
    snapshotfolder = Path(
        os.path.join(cfg["project_path"], str(modelfoldername), "train")
    )

    return trainposeconfigfile, testposeconfigfile, snapshotfolder

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    pretrained_model = vit_h_14(weights=weights)

    feature_extractor = ViTFeatureExtractor(
        image_size=pretrained_model.image_size,
        patch_size=pretrained_model.patch_size,
        num_layers=len(pretrained_model.encoder.layers),
        num_heads=pretrained_model.encoder.layers[0].num_heads,
        hidden_dim=pretrained_model.hidden_dim,
        mlp_dim=pretrained_model.mlp_dim
    )

    feature_extractor.load_state_dict(pretrained_model.state_dict(), strict=False)

    feature_extractor = feature_extractor.to(device)
    with torch.no_grad():
        image_path = '/data/home/athomas314/test_image.png'
        image = Image.open(image_path)
        extractor_input = torch.unsqueeze(weights.transforms()(image), 0)

        x = feature_extractor.forward(extractor_input)
    
    # TODO: figure out what to do with these (maybe make them part of a database or something)
    NUM_KPTS = 9 # figure out how to make this not hardcoded
    FEATURE_DIM = 2048 # figure out if there is a good place to put this
    img_channels = 3 
    n = 8
    
    in_shape = torch.Size([img_channels, pretrained_model.image_size, pretrained_model.image_size])
    out_shape = pretrained_model.hidden_dim
    
    config_path = '/data/home/athomas314/dlc_model-student-2023-07-26/config.yaml'
    shuffle = 1
    video_path = ['/data/home/athomas314/test_video/MC_singlenuc23_1_Tk33_0212200003_vid_clip_36170_38240_rotcrop.mp4']
    n_fish = 6
    n_triplets=1000
    train_epochs = 100
    trainingsetindex=0
    modelprefix=""
    train_frac=0.8
    
    cfg = auxiliaryfunctions.read_config(config_path)
    DLCscorer, _ = auxiliaryfunctions.GetScorerName(
        cfg,
        shuffle=shuffle,
        trainFraction=cfg["TrainingFraction"][trainingsetindex],
        modelprefix=modelprefix,
    )

    (
        _,
        _,
        snapshotfolder,
    ) = return_train_network_path(
        config_path,
        shuffle=shuffle,
        modelprefix=modelprefix,
        trainingsetindex=trainingsetindex,
    )
    
    train_tracking_transformer_image(config_path,
                               DLCscorer,
                               video_path,
                               videotype="",
                               train_frac=train_frac,
                               modelprefix=modelprefix,
                               n_triplets=n_triplets,
                               train_epochs=train_epochs,
                               batch_size=n,
                               ckpt_folder=snapshotfolder,
                               feature_dim=FEATURE_DIM,
                               num_kpts=NUM_KPTS,
                               feature_extractor=feature_extractor,
                               feature_extractor_in_dim=in_shape,
                               feature_extractor_out_dim=out_shape
                               )