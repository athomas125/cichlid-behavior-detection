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
"""
This is a modified version of the functions needed for create_tracking_dataset
in deeplabcut, so that I can modify the dataset being fed to the 
re-identification model to include crops of the relevant images.
"""


import os
import os.path
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import shelve

from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.pose_estimation_tensorflow.lib import trackingutils
from deeplabcut.refine_training_dataset.stitch import TrackletStitcher
from deeplabcut.pose_tracking_pytorch.tracking_utils.preprocessing import query_feature_by_coord_in_img_space

from deeplabcut.utils import auxiliaryfunctions, auxfun_models
from PIL import Image
from torchvision.models import ViT_H_14_Weights


def generate_train_triplets_from_pickle_for_vid(path_to_track, n_triplets=1000):
    ts = TrackletStitcher.from_pickle(path_to_track, 3)
    triplets = ts.mine(n_triplets)
    assert len(triplets) == n_triplets
    return triplets


def ViT_H_14_Preprocess(frame, weights):
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Apply the ViT transforms
    vit_input = weights.transforms()(pil_image)
    
    return vit_input

def save_train_triplets_with_img(feature_fname, 
                                 triplets,
                                 out_name,
                                 video_path,
                                 image_preprocess_func,
                                 image_preprocess_weights,
                                 OUTPUT_COLOR_CHANNELS=3):
    ret_vecs = []

    feature_dict = shelve.open(feature_fname, protocol=pickle.DEFAULT_PROTOCOL)

    nframes = len(feature_dict.keys())

    zfill_width = int(np.ceil(np.log10(nframes)))
    
    
    
    final_dim_size = image_preprocess_weights.transforms().crop_size[0]**2 * OUTPUT_COLOR_CHANNELS #squaring crop size cause square crop
    example = triplets[0][0]
    example_frame = "frame" + str(example[1]).zfill(zfill_width)
    kpt_feature_shape = query_feature_by_coord_in_img_space(
        feature_dict, example_frame, example[0]
    ).flatten().shape[0]
    
    
    # Initialize a memory-mapped array to avoid OOM issues
    ret_vecs_shape = (len(triplets), 3, kpt_feature_shape + final_dim_size)
    ret_vecs = np.memmap(out_name, dtype='float32', mode='w+', shape=ret_vecs_shape)
    
    frame_map = {}
    for i, triplet in enumerate(triplets):
        if i > 1:
            break
        anchor, pos, neg = triplet[0], triplet[1], triplet[2]

        anchor_coord, anchor_frame = anchor
        pos_coord, pos_frame = pos
        neg_coord, neg_frame = neg

        anchor_frame = "frame" + str(anchor_frame).zfill(zfill_width)
        pos_frame = "frame" + str(pos_frame).zfill(zfill_width)
        neg_frame = "frame" + str(neg_frame).zfill(zfill_width)

        if (
            anchor_frame in feature_dict
            and pos_frame in feature_dict
            and neg_frame in feature_dict
        ):
            # only try to find these features if they are in the dictionary

            anchor_vec = query_feature_by_coord_in_img_space(
                feature_dict, anchor_frame, anchor_coord
            ).flatten()
            pos_vec = query_feature_by_coord_in_img_space(
                feature_dict, pos_frame, pos_coord
            ).flatten()
            neg_vec = query_feature_by_coord_in_img_space(
                feature_dict, neg_frame, neg_coord
            ).flatten()

            ret_vecs[i, :, :kpt_feature_shape] = np.array([anchor_vec, pos_vec, neg_vec])
            triplet_ind = 0
            for coord, frame in [anchor, pos, neg]:
                frame_str = "frame" + str(frame).zfill(zfill_width)
                if frame_str not in frame_map:
                    frame_map[frame_str] = {}
                
                # using this so we have a hashable type (also using a mask here)
                bounding_coords = (max(coord[coord[:,0] >= 0,0]), min(coord[coord[:,0] >= 0,0]), max(coord[coord[:,1] >= 0,1]), min(coord[coord[:,1] >= 0,1]))
                if bounding_coords not in frame_map[frame_str]:
                    frame_map[frame_str][bounding_coords] = []
                frame_map[frame_str][bounding_coords].append((i, triplet_ind))
                triplet_ind += 1
                
    cap = cv2.VideoCapture(video_path)
    
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_ind = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_str = "frame" + str(frame_ind).zfill(zfill_width)
        if frame_str in frame_map:
            for key in frame_map[frame_str]:
                frame_crop = frame[max(key[3]-50, 0):min(key[2]+50, height),max(key[1]-50, 0):min(key[0]+50, width),:]
                vit_vec = image_preprocess_func(frame_crop, image_preprocess_weights).flatten()
                for i, ind in frame_map[frame_str][key]:
                    ret_vecs[i, ind, kpt_feature_shape:] = vit_vec
                    ret_vecs.flush() 
        frame_ind += 1
            
    # clear everything to disk
    ret_vecs.flush() 
    # close array
    del ret_vecs
        
def create_train_using_pickle_and_vid(feature_fname, path_to_pickle, out_name, video_path, n_triplets=1000,
                                      image_preprocess_func=None, image_preprocess_weights=None):
    triplets = generate_train_triplets_from_pickle_for_vid(
        path_to_pickle, n_triplets=n_triplets
    )
    save_train_triplets_with_img(feature_fname, triplets, out_name, video_path,
                                image_preprocess_func=image_preprocess_func, image_preprocess_weights=image_preprocess_weights,)
    

def create_triplets_dataset(
    videos,
    dlcscorer,
    track_method,
    n_triplets=1000,
    destfolder=None,
    image_preprocess_func=None,
    image_preprocess_weights=None,
):
    # 1) reference to video folder and get the proper bpt_feature file for feature table
    # 2) get either the path to gt or the path to track pickle

    for video in videos:
        vname = Path(video).stem
        videofolder = str(Path(video).parents[0])
        if destfolder is None:
            destfolder = videofolder
        feature_fname = os.path.join(
            destfolder, vname + dlcscorer + "_bpt_features.pickle"
        )

        method = trackingutils.TRACK_METHODS[track_method]
        track_file = os.path.join(destfolder, vname + dlcscorer + f"{method}.pickle")
        out_fname = os.path.join(destfolder, vname + dlcscorer + "_triplet_vector_with_image_debugging.npy")
        create_train_using_pickle_and_vid(
            feature_fname, track_file, out_fname, video, n_triplets=n_triplets,
            image_preprocess_func=image_preprocess_func, image_preprocess_weights=image_preprocess_weights,
        )

####################################################
# Loading data, and defining model folder
####################################################


def create_tracking_dataset(
    config,
    videos,
    track_method,
    videotype="",
    shuffle=1,
    trainingsetindex=0,
    gputouse=None,
    save_as_csv=False,
    destfolder=None,
    batchsize=None,
    cropping=None,
    TFGPUinference=True,
    dynamic=(False, 0.5, 10),
    modelprefix="",
    robust_nframes=False,
    n_triplets=1000,
    image_preprocess_func=None,
    image_preprocess_weights=None,
):
    # allow_growth must be true here because tensorflow does not automatically free gpu memory and setting it as false occupies all gpu memory so that pytorch cannot kick in
    allow_growth = True

    if "TF_CUDNN_USE_AUTOTUNE" in os.environ:
        del os.environ["TF_CUDNN_USE_AUTOTUNE"]  # was potentially set during training

    if gputouse is not None:  # gpu selection
        auxfun_models.set_visible_devices(gputouse)

    tf.compat.v1.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg["TrainingFraction"][trainingsetindex]

    if cropping is not None:
        cfg["cropping"] = True
        cfg["x1"], cfg["x2"], cfg["y1"], cfg["y2"] = cropping
        print("Overwriting cropping parameters:", cropping)
        print("These are used for all videos, but won't be save to the cfg file.")

    modelfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_model_folder(
                trainFraction, shuffle, cfg, modelprefix=modelprefix
            )
        ),
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist."
            % (shuffle, trainFraction)
        )

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [
                fn.split(".")[0]
                for fn in os.listdir(os.path.join(modelfolder, "train"))
                if "index" in fn
            ]
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle)
        )

    if cfg["snapshotindex"] == "all":
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running video analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!"
        )
        snapshotindex = -1
    else:
        snapshotindex = cfg["snapshotindex"]

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg["init_weights"] = os.path.join(
        modelfolder, "train", Snapshots[snapshotindex]
    )
    trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
    # Update number of output and batchsize
    dlc_cfg["num_outputs"] = cfg.get("num_outputs", dlc_cfg.get("num_outputs", 1))

    if batchsize is None:
        # update batchsize (based on parameters in config.yaml)
        dlc_cfg["batch_size"] = cfg["batch_size"]
    else:
        dlc_cfg["batch_size"] = batchsize
        cfg["batch_size"] = batchsize

    if "multi-animal" in dlc_cfg["dataset_type"]:
        dynamic = (False, 0.5, 10)  # setting dynamic mode to false
        TFGPUinference = False

    if dynamic[0]:  # state=true
        # (state,detectiontreshold,margin)=dynamic
        print("Starting analysis in dynamic cropping mode with parameters:", dynamic)
        dlc_cfg["num_outputs"] = 1
        TFGPUinference = False
        dlc_cfg["batch_size"] = 1
        print(
            "Switching batchsize to 1, num_outputs (per animal) to 1 and TFGPUinference to False (all these features are not supported in this mode)."
        )

    # Name for scorer:
    DLCscorer, DLCscorerlegacy = auxiliaryfunctions.get_scorer_name(
        cfg,
        shuffle,
        trainFraction,
        trainingsiterations=trainingsiterations,
        modelprefix=modelprefix,
    )
    if dlc_cfg["num_outputs"] > 1:
        if TFGPUinference:
            print(
                "Switching to numpy-based keypoint extraction code, as multiple point extraction is not supported by TF code currently."
            )
            TFGPUinference = False
        print("Extracting ", dlc_cfg["num_outputs"], "instances per bodypart")
        xyz_labs_orig = ["x", "y", "likelihood"]
        suffix = [str(s + 1) for s in range(dlc_cfg["num_outputs"])]
        suffix[0] = ""  # first one has empty suffix for backwards compatibility
        xyz_labs = [x + s for s in suffix for x in xyz_labs_orig]
    else:
        xyz_labs = ["x", "y", "likelihood"]

    if TFGPUinference:
        sess, inputs, outputs = predict.setup_GPUpose_prediction(
            dlc_cfg, allow_growth=allow_growth
        )
    else:
        sess, inputs, outputs, extra_dict = predict.setup_pose_prediction(
            dlc_cfg, allow_growth=allow_growth, collect_extra=True
        )

    pdindex = pd.MultiIndex.from_product(
        [[DLCscorer], dlc_cfg["all_joints_names"], xyz_labs],
        names=["scorer", "bodyparts", "coords"],
    )

    ##################################################
    # Looping over videos
    ##################################################
    Videos = auxiliaryfunctions.get_list_of_videos(videos, videotype)
    if len(Videos) > 0:
        if "multi-animal" in dlc_cfg["dataset_type"]:
            create_triplets_dataset(
                Videos,
                DLCscorer,
                track_method,
                n_triplets=n_triplets,
                destfolder=destfolder,
                image_preprocess_func=image_preprocess_func,
                image_preprocess_weights=image_preprocess_weights,
            )

        else:
            raise NotImplementedError("not implemented")

        os.chdir(str(start_path))
        if "multi-animal" in dlc_cfg["dataset_type"]:
            print(
                "If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames."
            )
        else:
            print(
                "The videos are analyzed. Now your research can truly start! \n You can create labeled videos with 'create_labeled_video'"
            )
            print(
                "If the tracking is not satisfactory for some videos, consider expanding the training set. You can use the function 'extract_outlier_frames' to extract a few representative outlier frames."
            )
        return DLCscorer  # note: this is either DLCscorer or DLCscorerlegacy depending on what was used!
    else:
        print("No video(s) were found. Please check your paths and/or 'videotype'.")
        return DLCscorer
    
if __name__ == "__main__":
    config = '/data/home/athomas314/dlc_model-student-2023-07-26/config.yaml'
    videos = ['/data/home/athomas314/test_video/MC_singlenuc23_1_Tk33_0212200003_vid_clip_36170_38240_rotcrop.mp4']
    track_method = 'ellipse'
    videotype= ""
    shuffle = 1
    trainingsetindex = 0
    modelprefix = ""
    n_triplets = 1000
    destfolder = None
    func = ViT_H_14_Preprocess
    weights = ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1
    
    create_tracking_dataset(
        config,
        videos,
        track_method,
        videotype=videotype,
        shuffle=shuffle,
        trainingsetindex=trainingsetindex,
        modelprefix=modelprefix,
        n_triplets=n_triplets,
        destfolder=destfolder,
        image_preprocess_func=func,
        image_preprocess_weights=weights,
    )