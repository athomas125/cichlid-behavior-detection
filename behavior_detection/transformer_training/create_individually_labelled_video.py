"""
This version has been tested with DEEPLABCUT 2.3.10
I DO NOT TAKE ANY CREDIT FOR THIS CODE - SEE THE LICENSE BELOW
This code has been adapted from the DEEPLABCUT CreateVideo script in utils/make_labeled_video.py
I created this so I have an easily accessible standalone version of the create video script that
has the ability to utilize the Semi-supervised transformer individual id labels generated from 
inference to label the given video.
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
from deeplabcut.modelzoo.utils import parse_available_supermodels
from skimage.draw import disk, line_aa, set_color
from skimage.util import img_as_ubyte
from tqdm import trange
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal, visualization
from deeplabcut.utils.video_processor import (
    VideoProcessorCV as vp,
)  # used to CreateVideo
from deeplabcut.utils.auxfun_videos import VideoWriter

# Code below this point is a modified version of make_labeled_video.py from Deeplabcut v2.3.10

def get_segment_indices(bodyparts2connect, all_bpts):
    bpts2connect = []
    for bpt1, bpt2 in bodyparts2connect:
        if bpt1 in all_bpts and bpt2 in all_bpts:
            bpts2connect.extend(
                zip(
                    *(
                        np.flatnonzero(all_bpts == bpt1),
                        np.flatnonzero(all_bpts == bpt2),
                    )
                )
            )
    return bpts2connect


def _get_default_conf_to_alpha(
    confidence_to_alpha: bool,
    pcutoff: float,
) -> Optional[Callable[[float], float]]:
    """Creates the default confidence_to_alpha function"""
    if not confidence_to_alpha:
        return None

    def default_confidence_to_alpha(x):
        if pcutoff == 0:
            return x
        return np.clip((x - pcutoff) / (1 - pcutoff), 0, 1)

    return default_confidence_to_alpha

def CreateVideo(
    clip,
    Dataframe,
    pcutoff,
    dotsize,
    colormap,
    bodyparts2plot,
    trailpoints,
    cropping,
    x1,
    x2,
    y1,
    y2,
    bodyparts2connect,
    skeleton_color,
    draw_skeleton,
    displaycropped,
    color_by,
    confidence_to_alpha=None,
    plot_with_center_trail=True,
    center_trailpoints=90
):
    """Creating individual frames with labeled body parts and making a video"""
    bpts = Dataframe.columns.get_level_values("bodyparts")
    all_bpts = bpts.values[::3]
    if draw_skeleton:
        color_for_skeleton = (
            np.array(mcolors.to_rgba(skeleton_color))[:3] * 255
        ).astype(np.uint8)
        # recode the bodyparts2connect into indices for df_x and df_y for speed
        bpts2connect = get_segment_indices(bodyparts2connect, all_bpts)

    if displaycropped:
        ny, nx = y2 - y1, x2 - x1
    else:
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

    df_x, df_y, df_likelihood = Dataframe.values.reshape((len(Dataframe), -1, 3)).T

    if cropping and not displaycropped:
        df_x += x1
        df_y += y1
    colorclass = plt.cm.ScalarMappable(cmap=colormap)

    bplist = bpts.unique().to_list()
    nbodyparts = len(bplist)
    if Dataframe.columns.nlevels == 3:
        nindividuals = int(len(all_bpts) / len(set(all_bpts)))
        map2bp = list(np.repeat(list(range(len(set(all_bpts)))), nindividuals))
        map2id = list(range(nindividuals)) * len(set(all_bpts))
    else:
        nindividuals = len(Dataframe.columns.get_level_values("individuals").unique())
        map2bp = [bplist.index(bp) for bp in all_bpts]
        nbpts_per_ind = (
            Dataframe.groupby(level="individuals", axis=1).size().values // 3
        )
        map2id = []
        for i, j in enumerate(nbpts_per_ind):
            map2id.extend([i] * j)
    keep = np.flatnonzero(np.isin(all_bpts, bodyparts2plot))
    bpts2color = [(ind, map2bp[ind], map2id[ind]) for ind in keep]

    if color_by == "bodypart":
        C = colorclass.to_rgba(np.linspace(0, 1, nbodyparts))
    else:
        C = colorclass.to_rgba(np.linspace(0, 1, nindividuals))
    colors = (C[:, :3] * 255).astype(np.uint8)

    with np.errstate(invalid="ignore"):
        center_points = [[] for _ in range(nindividuals)]  # Initialize list to store center points
        
        for index in trange(min(nframes, len(Dataframe))):
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
                    
            if plot_with_center_trail:
                for individual in range(nindividuals):
                    individual_indices = [i for i, x in enumerate(map2id) if x == individual]
                    avg_x = np.mean(df_x[individual_indices, index])
                    avg_y = np.mean(df_y[individual_indices, index])
                    # center point
                    rr, cc = disk((avg_y, avg_x), dotsize, shape=(ny, nx))
                    set_color(image, (rr, cc), colors[individual], 1)  # Same color for center point
                    # creating trail
                    center_points[individual].append((avg_y, avg_x))
                    for k in range(1, min(center_trailpoints, len(center_points[individual]))):
                        rr, cc = disk(center_points[individual][-k], dotsize, shape=(ny, nx))
                        set_color(image, (rr, cc), colors[individual], 1)  # Same color for center trail


            clip.save_frame(image)
    clip.close()
    
def _create_labeled_video(
    video,
    h5file,
    keypoints2show="all",
    animals2show="all",
    skeleton_edges=None,
    pcutoff=0.6,
    dotsize=8,
    cmap="cool",
    color_by="bodypart",
    skeleton_color="k",
    trailpoints=0,
    bbox=None,
    display_cropped=False,
    codec="mp4v",
    fps=None,
    output_path="",
    confidence_to_alpha=None,
    plot_with_center_trail=True,
    center_trailpoints=90
):
    if color_by not in ("bodypart", "individual"):
        raise ValueError("`color_by` should be either 'bodypart' or 'individual'.")

    if not output_path:
        s = "_id" if color_by == "individual" else "_bp"
        output_path = h5file.replace(".h5", f"{s}_labeled.mp4")

    x1, x2, y1, y2 = bbox
    if display_cropped:
        sw = x2 - x1
        sh = y2 - y1
    else:
        sw = sh = ""

    clip = vp(
        fname=video,
        sname=output_path,
        codec=codec,
        sw=sw,
        sh=sh,
        fps=fps,
    )
    cropping = bbox != (0, clip.w, 0, clip.h)
    df = pd.read_hdf(h5file)
    try:
        animals = df.columns.get_level_values("individuals").unique().to_list()
        if animals2show != "all" and isinstance(animals, Iterable):
            animals = [a for a in animals if a in animals2show]
        df = df.loc(axis=1)[:, animals]
    except KeyError:
        pass
    kpts = df.columns.get_level_values("bodyparts").unique().to_list()
    if keypoints2show != "all" and isinstance(keypoints2show, Iterable):
        kpts = [kpt for kpt in kpts if kpt in keypoints2show]
    CreateVideo(
        clip,
        df,
        pcutoff,
        dotsize,
        cmap,
        kpts,
        trailpoints,
        cropping,
        x1,
        x2,
        y1,
        y2,
        skeleton_edges,
        skeleton_color,
        bool(skeleton_edges),
        display_cropped,
        color_by,
        confidence_to_alpha=confidence_to_alpha,
        plot_with_center_trail=plot_with_center_trail,
        center_trailpoints=center_trailpoints
    )

    
def proc_video(
    videos,
    destfolder,
    filtered,
    DLCscorer,
    DLCscorerlegacy,
    track_method,
    cfg,
    individuals,
    color_by,
    bodyparts,
    codec,
    bodyparts2connect,
    trailpoints,
    outputframerate,
    draw_skeleton,
    skeleton_color,
    displaycropped,
    overwrite,
    video,
    init_weights="",
    confidence_to_alpha: Optional[Callable[[float], float]] = None,
    use_transf_labels=False,
    plot_with_center_trail=False,
    center_trailpoints=0,
):
    """Helper function for create_videos

    Parameters
    ----------


    Returns
    -------
        result : bool
        ``True`` if a video is successfully created.
    """
    videofolder = Path(video).parents[0]
    if destfolder is None:
        destfolder = videofolder  # where your folder with videos is.

    auxiliaryfunctions.attempt_to_make_folder(destfolder)

    os.chdir(destfolder)  # THE VIDEO IS STILL IN THE VIDEO FOLDER
    print("Starting to process video: {}".format(video))
    vname = str(Path(video).stem)

    if init_weights != "":
        DLCscorer = "DLC_" + Path(init_weights).stem
        DLCscorerlegacy = "DLC_" + Path(init_weights).stem

    if filtered:
        videooutname1 = os.path.join(vname + DLCscorer + "filtered_labeled.mp4")
        videooutname2 = os.path.join(vname + DLCscorerlegacy + "filtered_labeled.mp4")
    else:
        videooutname1 = os.path.join(vname + DLCscorer + "_labeled.mp4")
        videooutname2 = os.path.join(vname + DLCscorerlegacy + "_labeled.mp4")

    if (
        os.path.isfile(videooutname1) or os.path.isfile(videooutname2)
    ) and not overwrite:
        print("Labeled video {} already created.".format(vname))
        return True
    else:
        print("Loading {} and data.".format(video))
        try:
            df, filepath, _, _ = auxiliaryfunctions.load_analyzed_data(
                destfolder, vname, DLCscorer, filtered, track_method
            )
            metadata = auxiliaryfunctions.load_video_metadata(
                destfolder, vname, DLCscorer
            )
            if cfg.get("multianimalproject", False):
                s = "_id" if color_by == "individual" else "_bp"
            else:
                s = ""
            if use_transf_labels:
                filepath.replace("_filtered", "_tr")
            videooutname = filepath.replace(".h5", f"{s}_labeled.mp4")
            if os.path.isfile(videooutname) and not overwrite:
                print("Labeled video already created. Skipping...")
                return
            
            if all(individuals):
                df = df.loc(axis=1)[:, df.columns.get_level_values("individuals").unique()[:]]
            cropping = metadata["data"]["cropping"]
            [x1, x2, y1, y2] = metadata["data"]["cropping_parameters"]
            labeled_bpts = [
                bp
                for bp in df.columns.get_level_values("bodyparts").unique()
                if bp in bodyparts
            ]
            
            _create_labeled_video(
                video,
                filepath,
                keypoints2show=labeled_bpts,
                animals2show=df.columns.get_level_values("individuals").unique().to_list(),
                bbox=(x1, x2, y1, y2),
                codec=codec,
                output_path=videooutname,
                pcutoff=cfg["pcutoff"],
                dotsize=cfg["dotsize"],
                cmap=cfg["colormap"],
                color_by=color_by,
                skeleton_edges=bodyparts2connect,
                skeleton_color=skeleton_color,
                trailpoints=trailpoints,
                fps=outputframerate,
                display_cropped=displaycropped,
                confidence_to_alpha=confidence_to_alpha,
                plot_with_center_trail=plot_with_center_trail,
                center_trailpoints=center_trailpoints,
            )
            return True

        except FileNotFoundError as e:
            print(e)
            return False


def create_labeled_video_w_transformer_option(
    config,
    videos,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    filtered=False,
    displayedbodyparts="all",
    displayedindividuals="all",
    codec="mp4v",
    outputframerate=None,
    destfolder=None,
    draw_skeleton=False,
    trailpoints=0,
    displaycropped=False,
    color_by="bodypart",
    modelprefix="",
    init_weights="",
    track_method="",
    superanimal_name="",
    pcutoff=0.6,
    skeleton=[],
    skeleton_color="white",
    dotsize=8,
    colormap="rainbow",
    alphavalue=0.5,
    overwrite=False,
    confidence_to_alpha: Union[bool, Callable[[float], float]] = False,
    use_transf_labels=False,
    plot_with_center_trail=False,
    center_trailpoints=0,
):
    """Labels the bodyparts in a video.

    Make sure the video is already analyzed by the function
    ``deeplabcut.analyze_videos``.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file.

    videos : list[str]
        A list of strings containing the full paths to videos for analysis or a path
        to the directory, where all the videos with same extension are stored.

    videotype: str, optional, default=""
        Checks for the extension of the video in case the input to the video is a
        directory. Only videos with this extension are analyzed.
        If left unspecified, videos with common extensions
        ('avi', 'mp4', 'mov', 'mpeg', 'mkv') are kept.

    shuffle : int, optional, default=1
        Number of shuffles of training dataset.

    trainingsetindex: int, optional, default=0
        Integer specifying which TrainingsetFraction to use.
        Note that TrainingFraction is a list in config.yaml.

    filtered: bool, optional, default=False
        Boolean variable indicating if filtered output should be plotted rather than
        frame-by-frame predictions. Filtered version can be calculated with
        ``deeplabcut.filterpredictions``.

    displayedbodyparts: list[str] or str, optional, default="all"
        This selects the body parts that are plotted in the video. If ``all``, then all
        body parts from config.yaml are used. If a list of strings that are a subset of
        the full list. E.g. ['hand','Joystick'] for the demo
        Reaching-Mackenzie-2018-08-30/config.yaml to select only these body parts.

    displayedindividuals: list[str] or str, optional, default="all"
        Individuals plotted in the video.
        By default, all individuals present in the config will be showed.

    codec: str, optional, default="mp4v"
        Codec for labeled video. For available options, see
        http://www.fourcc.org/codecs.php. Note that this depends on your ffmpeg
        installation.

    outputframerate: int or None, optional, default=None
        Positive number, output frame rate for labeled video (only available for the
        mode with saving frames.) If ``None``, which results in the original video
        rate.

    destfolder: string or None, optional, default=None
        Specifies the destination folder that was used for storing analysis data. If
        ``None``, the path of the video file is used.

    draw_skeleton: bool, optional, default=False
        If ``True`` adds a line connecting the body parts making a skeleton on each
        frame. The body parts to be connected and the color of these connecting lines
        are specified in the config file.

    trailpoints: int, optional, default=0
        Number of previous frames whose body parts are plotted in a frame
        (for displaying history).

    displaycropped: bool, optional, default=False
        Specifies whether only cropped frame is displayed (with labels analyzed
        therein), or the original frame with the labels analyzed in the cropped subset.

    color_by : string, optional, default='bodypart'
        Coloring rule. By default, each bodypart is colored differently.
        If set to 'individual', points belonging to a single individual are colored the
        same.

    modelprefix: str, optional, default=""
        Directory containing the deeplabcut models to use when evaluating the network.
        By default, the models are assumed to exist in the project folder.

    init_weights: str,
        Checkpoint path to the super model

    track_method: string, optional, default=""
        Specifies the tracker used to generate the data.
        Empty by default (corresponding to a single animal project).
        For multiple animals, must be either 'box', 'skeleton', or 'ellipse' and will
        be taken from the config.yaml file if none is given.

    overwrite: bool, optional, default=False
        If ``True`` overwrites existing labeled videos.

    confidence_to_alpha: Union[bool, Callable[[float], float], default=False
        If False, all keypoints will be plot with alpha=1. Otherwise, this can be
        defined as a function f: [0, 1] -> [0, 1] such that the alpha value for a
        keypoint will be set as a function of its score: alpha = f(score). The default
        function used when True is f(x) = max(0, (x - pcutoff)/(1 - pcutoff)).

    Returns
    -------
        results : list[bool]
        ``True`` if the video is successfully created for each item in ``videos``.

    Examples
    --------

    Create the labeled video for a single video

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/reachingvideo1.avi'],
        )

    Create the labeled video for multiple videos

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            [
                '/analysis/project/videos/reachingvideo1.avi',
                '/analysis/project/videos/reachingvideo2.avi',
            ],
        )

    Create the labeled video for all the videos with an .avi extension in a directory.

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/'],
        )

    Create the labeled video for all the videos with an .mp4 extension in a directory.

    >>> deeplabcut.create_labeled_video(
            '/analysis/project/reaching-task/config.yaml',
            ['/analysis/project/videos/'],
            videotype='mp4',
        )
    """
    if config == "":
        pass
    else:
        cfg = auxiliaryfunctions.read_config(config)
        trainFraction = cfg["TrainingFraction"][trainingsetindex]
        track_method = auxfun_multianimal.get_track_method(
            cfg, track_method=track_method
        )

    if init_weights == "":
        DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
            cfg, shuffle, trainFraction, modelprefix=modelprefix
        )  # automatically loads corresponding model (even training iteration based on snapshot index)
    else:
        DLCscorer = "DLC_" + Path(init_weights).stem
        DLCscorerlegacy = "DLC_" + Path(init_weights).stem

    # parse the alpha selection function
    if isinstance(confidence_to_alpha, bool):
        confidence_to_alpha = _get_default_conf_to_alpha(confidence_to_alpha, pcutoff)

    if superanimal_name != "":
        dlc_root_path = auxiliaryfunctions.get_deeplabcut_path()
        supermodels = parse_available_supermodels()
        test_cfg = load_config(
            os.path.join(
                dlc_root_path,
                "pose_estimation_tensorflow",
                "superanimal_configs",
                supermodels[superanimal_name],
            )
        )

        bodyparts = test_cfg["all_joints_names"]
        cfg = {
            "skeleton": skeleton,
            "skeleton_color": skeleton_color,
            "pcutoff": pcutoff,
            "dotsize": dotsize,
            "alphavalue": alphavalue,
            "colormap": colormap,
        }
    else:
        bodyparts = (
            auxiliaryfunctions.intersection_of_body_parts_and_ones_given_by_user(
                cfg, displayedbodyparts
            )
        )

    individuals = auxfun_multianimal.IntersectionofIndividualsandOnesGivenbyUser(
        cfg, displayedindividuals
    )
    if draw_skeleton:
        bodyparts2connect = cfg["skeleton"]
        if displayedbodyparts != "all":
            bodyparts2connect = [
                pair
                for pair in bodyparts2connect
                if all(element in displayedbodyparts for element in pair)
            ]
        skeleton_color = cfg["skeleton_color"]
    else:
        bodyparts2connect = None
        skeleton_color = None

    start_path = os.getcwd()
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)

    if not Videos:
        return []

    func = partial(
        proc_video,
        videos,
        destfolder,
        filtered,
        DLCscorer,
        DLCscorerlegacy,
        track_method,
        cfg,
        individuals,
        color_by,
        bodyparts,
        codec,
        bodyparts2connect,
        trailpoints,
        outputframerate,
        draw_skeleton,
        skeleton_color,
        displaycropped,
        overwrite,
        init_weights=init_weights,
        confidence_to_alpha=confidence_to_alpha,
        use_transf_labels=use_transf_labels,
        plot_with_center_trail=plot_with_center_trail,
        center_trailpoints=center_trailpoints,
    )
    
    if get_start_method() == "fork" and len(Videos) > 1:
        with Pool(min(os.cpu_count(), len(Videos))) as pool:
            results = pool.map(func, Videos)
    else:
        results = []
        for video in Videos:
            results.append(func(video))

    os.chdir(start_path)
    return results
