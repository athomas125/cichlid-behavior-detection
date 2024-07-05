import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the repository root directory to sys.path
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
fs_root = os.path.abspath(os.path.join(current_dir, '../../..'))
if fs_root not in sys.path:
    sys.path.insert(0, fs_root)
    
import deeplabcut as dlc
from behavior_detection.misc.analyse_videos import analyse_videos
from behavior_detection.misc.train_network import kill_and_reset
from behavior_detection.transformer_training.create_individually_labelled_video import create_labeled_video_w_transformer_option
from CichlidBowerTracking.ImageProcessing.src.crop_and_rotate_video import crop_and_rotate_video

def main():
    config_path = '/data/home/athomas314/dlc_model-student-2023-07-26/config.yaml'
    shuffle = 1
    
    # config_path = '/data/home/athomas314/dlc_model-student-2023-07-26_cropped/config.yaml'

    # for testing zero shot
    # config_path = '/data/home/athomas314/dlc_model-student-zero-shot-yellowhead/config.yaml'

    # config_path = '/data/home/athomas314/YellowHeadCroppedVid1-Adam-2024-05-29/config.yaml'
    
    # dlc.create_multianimaltraining_dataset(config_path)
     
    # train_pose_config, _, _ = dlc.return_train_network_path(config_path)
    # augs = {
    #     "gaussian_noise": True,
    #     "elastic_transform": True,
    #     "rotation": 180,
    #     "motion_blur": True,
    # }
    # dlc.auxiliaryfunctions.edit_config(
    #     train_pose_config,
    #     augs,
    # )

    # dlc.train_network(config_path, shuffle=shuffle, gputouse=1, saveiters=10000, maxiters=100000)
    # dlc.evaluate_network(config_path, gputouse=2)
    
    video_path = ['/data/home/athomas314/test_video/MC_singlenuc23_1_Tk33_0212200003_vid_clip_36170_38240_rotcrop.mp4']
    n_fish = 6
    # # video_path = ['/data/home/athomas314/oneshot_videos/MC_singlenuc23_1_Tk33_0212200003_vid_debugging_clip.mp4',
    # #               '/data/home/athomas314/oneshot_videos/MC_singlenuc34_3_Tk43_0303200001_vid_clip_1240_1360.mp4']
    # testing_long_video = True
    # if testing_long_video:
    #     video_path = ['/data/home/athomas314/long_video/0003_vid.mp4']
    #     crop_and_rotate = False
    #     if crop_and_rotate:
    #         video_path[0] = crop_and_rotate_video(video_path[0], 25, 87, 89, 1112, 914) # specific parameters for singlenuc23_1_Tk33
    #     else:
    #         video_path = ['/data/home/athomas314/long_video/0003_vid_rotcrop.mp4']
    # for video in video_path:
    # #     print(f"analyzing {video}")
    # #     # try:
    #     analyse_videos(config_path=config_path, videos=[video], shuffle=shuffle, n_fish=n_fish, gpu_to_use=7)
        # except Exception:
        #     kill_and_reset()
        #     continue

    # # # Analyse videos must be run before doing transformer reidentification and the transformer re-id 
    # # # uses the bodypart embeddings (generated by analyse_videos) + tracks as it's inputs.
    train_epochs=100 #100
    (_, _, snapshot_folder) =  dlc.return_train_network_path(config_path,
        shuffle=shuffle,
        modelprefix="",
        trainingsetindex=0,
    )

    transformer_checkpoint = os.path.join(
        snapshot_folder, f"dlc_transreid_{train_epochs}.pth"
    )

    # if not os.path.exists(transformer_checkpoint):
    dlc.transformer_reID(config_path, 
                         video_path,
                         n_tracks=n_fish,
                         n_triplets=100000,
                         train_epochs=train_epochs)
    # else:
    #     dlc.stitch_tracklets(
    #                         config_path,
    #                         video_path,
    #                         videotype="",
    #                         shuffle=shuffle,
    #                         trainingsetindex=0,
    #                         track_method="ellipse",
    #                         n_tracks=n_fish,
    #                         transformer_checkpoint=transformer_checkpoint,
    #                     )
    
    create_labeled_video_w_transformer_option(config_path, 
                                            video_path, 
                                            shuffle=shuffle,
                                            filtered=True,
                                            color_by="individual",
                                            overwrite=True,
                                            use_transf_labels=True,
                                            plot_with_center_trail=True,
                                            center_trailpoints=90
                                            )
    # dlc.plot_trajectories(config_path, video_path, shuffle=shuffle)
    
if __name__ == "__main__":
    main()