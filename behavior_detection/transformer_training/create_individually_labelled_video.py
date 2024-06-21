"""
This code has been adapted from the DEEPLABCUT CreateVideo script in utils/make_labeled_video.py
"""


import argparse
import os

import os.path
from pathlib import Path
from functools import partial
from multiprocessing import Pool, get_start_method
from typing import Iterable, Callable, List, Optional, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from skimage.draw import disk, line_aa, set_color
from skimage.util import img_as_ubyte
from tqdm import trange
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, visualization
from deeplabcut.utils.video_processor import (
    VideoProcessorCV as vp,
)  # used to CreateVideo
from deeplabcut.utils.auxfun_videos import VideoWriter


# TODO use this to not hardcode the tracks file name
video_folders = ['/data/home/athomas314/test_video/']

# TODO make this not hardcoded
video = '/data/home/athomas314/test_video/MC_singlenuc23_1_Tk33_0212200003_vid_clip_36170_38240.mp4'

transformer_id_tracks_file = '/data/home/athomas314/test_video/MC_singlenuc23_1_Tk33_0212200003_vid_clip_36170_38240DLC_dlcrnetms5_dlc_modelJul26shuffle1_100000_el_tr.h5'
tracks_df = pd.read_hdf(transformer_id_tracks_file)

animals = tracks_df.columns.get_level_values("individuals").unique().to_list()
tracks_df = tracks_df.loc(axis=1)[:, animals]

bpts = tracks_df.columns.get_level_values("bodyparts")
all_bpts = bpts.values[::3]
bplist = bpts.unique().to_list()
nbodyparts = len(bplist)
nindividuals = len(animals)


output_path = transformer_id_tracks_file.replace(".h5", "_tf_labeled.mp4")
codec = "mp4v" #default 
sw = sh = ""
fps = 30 # frame rate of the video
display_cropped = False # video not cropped

clip = vp(
        fname=video,
        sname=output_path,
        codec=codec,
        sw=sw,
        sh=sh,
        fps=fps,
    )

ny, nx = clip.height(), clip.width()

fps = clip.fps()
if isinstance(fps, float):
    if fps * 1000 > 65535:
        fps = round(fps)
nframes = clip.nframes
duration = nframes / fps

print(
    "Duration of video [s]: {}, recorded with {} fps!".format(
        round(duration, 2), round(fps, 2)
    )
)
print(
    "Overall # of frames: {} with cropped frame dimensions: {} {}".format(
        nframes, nx, ny
    )
)
print("Generating frames and creating video.")

df_x, df_y, df_likelihood = tracks_df.values.reshape((len(tracks_df), -1, 3)).T

# if cropping and not displaycropped:
#     df_x += x1
#     df_y += y1
colorclass = plt.cm.ScalarMappable(cmap=colormap)

# bplist = bpts.unique().to_list()
# nbodyparts = len(bplist)
# if Dataframe.columns.nlevels == 3:
#     nindividuals = int(len(all_bpts) / len(set(all_bpts)))
#     map2bp = list(np.repeat(list(range(len(set(all_bpts)))), nindividuals))
#     map2id = list(range(nindividuals)) * len(set(all_bpts))
# else:
#     nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
#     map2bp = [bplist.index(bp) for bp in all_bpts]
#     nbpts_per_ind = (
#         Dataframe.groupby(level="individuals", axis=1).size().values // 3
#     )
#     map2id = []
#     for i, j in enumerate(nbpts_per_ind):
#         map2id.extend([i] * j)
# keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
# bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]

# # if color_by == "bodypart":
# #     C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
# # else:
# C = colorclass.to_rgba(np.linspace(0, 1, nindividuals))
# colors = (C[:, :3] * 255).astype(np.uint8)

with np.errstate(invalid="ignore"):
    for index in trange(min(nframes, len(tracks_df))):
        image = clip.load_frame()
        if displaycropped:
            image = image[y1:y2, x1:x2]

        # Draw the skeleton for specific bodyparts to be connected as
        # specified in the config file
        if draw_skeleton:
            for bpt1, bpt2 in bpts2connect:
                if np.all(df_likelihood[[bpt1, bpt2], index] > pcutoff) and not (
                    np.any(np.isnan(df_x[[bpt1, bpt2], index]))
                    or np.any(np.isnan(df_y[[bpt1, bpt2], index]))
                ):
                    rr, cc, val = line_aa(
                        int(np.clip(df_y[bpt1, index], 0, ny - 1)),
                        int(np.clip(df_x[bpt1, index], 0, nx - 1)),
                        int(np.clip(df_y[bpt2, index], 1, ny - 1)),
                        int(np.clip(df_x[bpt2, index], 1, nx - 1)),
                    )
                    image[rr, cc] = color_for_skeleton

        for ind, num_bp, num_ind in bpts2color:
            if df_likelihood[ind, index] > pcutoff:
                if color_by == "bodypart":
                    color = colors[num_bp]
                else:
                    color = colors[num_ind]
                if trailpoints > 0:
                    for k in range(1, min(trailpoints, index + 1)):
                        rr, cc = disk(
                            (df_y[ind, index - k], df_x[ind, index - k]),
                            dotsize,
                            shape=(ny, nx),
                        )
                        image[rr, cc] = color
                rr, cc = disk(
                    (df_y[ind, index], df_x[ind, index]), dotsize, shape=(ny, nx)
                )
                alpha = 1
                if confidence_to_alpha is not None:
                    alpha = confidence_to_alpha(df_likelihood[ind, index])

                set_color(image, (rr, cc), color, alpha)

        clip.save_frame(image)
clip.close()