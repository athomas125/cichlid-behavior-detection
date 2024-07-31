"""
This is a modification of the make_model.py file in deeplabcut pose_tracking_pytorch/model 
This modification was made so that the model could also process a cropped image with the 
animal in frame using a model as a feature extractor to do a preprocessing step on the
cropped image.
"""
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

import torch
import torch.nn as nn
from deeplabcut.pose_tracking_pytorch.model.make_model import build_dlc_transformer
from deeplabcut.pose_tracking_pytorch.model.backbones.vit_pytorch import dlc_base_kpt_TransReID


class build_dlc_transformer_just_feat_extractor(build_dlc_transformer):
    """
    A class that builds a DLC transformer with an additional feature extractor.

    This class extends the build_dlc_transformer class by incorporating a feature extractor
    before passing the data through the transformer. It concatenates the extracted features
    with the original input before forwarding it to the parent class.

    Parameters:
    -----------
    cfg : object
        Configuration object containing model parameters.
    in_chans : int
        Number of input channels for non-image data.
    kpt_num : int
        Number of keypoints.
    factory : object
        Factory object for creating model components.
    feature_extractor : nn.Module
        The feature extractor module to be used.
    feature_extractor_in_shape : tuple
        The input shape expected by the feature extractor.
    feature_extractor_out_dim : int
        The output dimension of the feature extractor.

    Attributes:
    -----------
    non_image_data_chans : int
        Number of channels for non-image data.
    feature_extractor_in_shape : tuple
        The input shape expected by the feature extractor.
    feature_extractor : nn.Module
        The feature extractor module.

    Methods:
    --------
    forward(x: torch.Tensor) -> torch.Tensor:
        Performs a forward pass through the network.
    
    load_param(trained_path: str) -> None:
        Loads pretrained parameters from a given path.
    """
    def __init__(self, cfg, in_chans, kpt_num, factory, feature_extractor, feature_extractor_in_shape, feature_extractor_out_dim):
        
        self.kpt_num = kpt_num
        self.in_chans = in_chans
        self.non_image_data_chans = in_chans * kpt_num
        super().__init__(cfg, in_chans, kpt_num, factory)
        self.feature_extractor_in_shape = feature_extractor_in_shape
        self.feature_extractor = feature_extractor
        self.final_fc = nn.Linear(feature_extractor_out_dim, 1024)

    def forward(self, x):
        # base_input = torch.reshape(x[:, :self.non_image_data_chans], (x.shape[0], self.kpt_num, self.in_chans))
        # base_embeds = super().forward(base_input)
        images = torch.reshape(x[:,self.non_image_data_chans:], (-1, *self.feature_extractor_in_shape))
        image_embeddings = self.feature_extractor.forward(images)
        return self.final_fc(image_embeddings)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))


__factory_T_type = {
    "dlc_transreid": dlc_base_kpt_TransReID,
}


def make_dlc_model_just_image(cfg, feature_dim, kpt_num, feature_extractor=None, feature_extractor_in_dim=0, feature_extractor_out_dim=0):
    if feature_extractor is None:
        model = build_dlc_transformer(cfg, feature_dim, kpt_num, __factory_T_type)
    else:
        model = build_dlc_transformer_just_feat_extractor(cfg,
                                                    feature_dim,
                                                    kpt_num,
                                                    __factory_T_type,
                                                    feature_extractor,
                                                    feature_extractor_in_dim,
                                                    feature_extractor_out_dim)

    return model