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


class build_dlc_transformer_w_feat_extractor(build_dlc_transformer):
    def __init__(self, cfg, in_chans, kpt_num, factory, feature_extractor, feature_extractor_out_dim):
        self.non_image_data_chans = in_chans
        in_chans = in_chans + feature_extractor_out_dim
        super().__init__(cfg, in_chans, kpt_num, factory)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        base_input = x[:,:self.non_image_data_chans, :, :]
        ext_feat = self.feature_extractor(x[:,self.non_image_data_chans:, :, :])
        x = torch.cat([base_input, ext_feat], dim=1)
        
        return super().forward(x)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace("module.", "")].copy_(param_dict[i])
        print("Loading pretrained model from {}".format(trained_path))


__factory_T_type = {
    "dlc_transreid": dlc_base_kpt_TransReID,
}


def make_dlc_model(cfg, feature_dim, kpt_num, feature_extractor=None, feature_extractor_out_dim=0):
    model = build_dlc_transformer(cfg, feature_dim, kpt_num, __factory_T_type, feature_extractor, feature_extractor_out_dim)

    return model
