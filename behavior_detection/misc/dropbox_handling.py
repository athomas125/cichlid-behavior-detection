import os
import subprocess

from behavior_detection.BehavioralVideo import BehavioralVideo


def upload_to_dropbox(local_folder, remote_folder):
    path = local_folder
    for trial in os.listdir(local_folder):
        bc_clips = os.path.join(path, trial, "bower-circling-clips")
        if not os.path.exists(bc_clips):
            print(f"No bower circling clips found in {trial}")
            continue

        remote_path = os.path.join(remote_folder, trial)
        result = subprocess.run(['rclone', 'lsf', 'dropbox:' + remote_path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Folder '{trial}' already exists in Dropbox.")
        else:
            print(f"Creating folder '{trial}' in Dropbox...")
            create_folder_cmd = ['rclone', 'mkdir', 'dropbox:' + remote_path]
            subprocess.run(create_folder_cmd)

        print(f"Uploading {trial} to Dropbox...")
        subprocess.run(['rclone', 'copy', bc_clips, 'dropbox:' + remote_path])


def find_trial_vid(items):
    for item in items:
        if item.endswith(".mp4"):
            return item
    return None


def find_tracklets_file(items):
    for item in items:
        if item.endswith("filtered.csv"):
            return item
    return None


def get_recent_video(dropbox_folder):
    import json
    # Run the rclone command to list files in the Dropbox folder in JSON format
    command = ['rclone', 'lsjson', 'dropbox:' + dropbox_folder]
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Parse the JSON output
        files = json.loads(result.stdout)

        # Filter video files
        video_files = [file for file in files if file['MimeType'].startswith('video/')]

        if len(video_files) == 1:
            # If there's only one video, return its absolute path
            return os.path.join(dropbox_folder, video_files[0]['Path'])
        elif len(video_files) > 1:
            # If there are multiple videos, return the most recently modified one's absolute path
            video_files.sort(key=lambda x: x['ModTime'], reverse=True)
            return os.path.join(dropbox_folder, video_files[0]['Path'])
        else:
            print("No video files found.")
            return None
    else:
        # Handle error
        print("Error occurred while listing files.")
        return None


def get_clips(trial, config):
    items = os.listdir(trial)
    vid = find_trial_vid(items)
    if vid is None:
        # no video for this trial. download it from dropbox
        import subprocess
        path = "BioSci-McGrath/Apps/CichlidPiData/__ProjectData/Single_nuc_1/"
        path = os.path.join(path, os.path.basename(trial), "Videos")
        vid_path = get_recent_video(path)
        subprocess.run(['rclone', 'copy', 'dropbox:' + vid_path, trial])
    else:
        vid = os.path.join(trial, vid)
    tracklets = os.path.join(trial, find_tracklets_file(items))
    vid = BehavioralVideo(video_path=vid, config=config, shuffle=4, tracklets_path=tracklets)
    vid.calculate_velocities()
    vid.check_bower_circling(threshold=120, extract_clips=True, bower_circling_length=32)


def get_clips_from_clustering_data(trial, behavior, p_cutoff=0.7):
    # behavior in {'x', 'f', 'd', 'm', 's', 'o', 't', 'c'}
    # only care about 's' - spawning at the moment
    import pandas as pd
    from datetime import timedelta
    from behavior_detection.misc.ffmpeg_split import get_video_length
    from behavior_detection.misc.video_auxiliary import get_video_fps

    # download clustering data from dropbox if it doesn't exist
    clustering = os.path.join(trial, "clustering")
    if not os.path.isdir(clustering):
        os.mkdir(clustering)
    clusters_csv = os.path.join(clustering, "AllLabeledClusters.csv")
    if not os.path.exists(clusters_csv):
        trial_id = os.path.basename(trial)
        path = "BioSci-McGrath/Apps/CichlidPidata/__ProjectData/Single_nuc_1/"
        path = os.path.join(path, trial_id, "MasterAnalysisFiles/AllLabeledClusters.csv")
        subprocess.run(['rclone', 'copy', 'dropbox:' + path, clustering])
    vid = find_trial_vid(os.listdir(trial))
    frame_rate = get_video_fps(os.path.join(trial, vid))
    t_delta = 2.5
    length = 5  # 5 second clips


    # read clustering data
    clusters = pd.read_csv(clusters_csv)
    clusters = clusters[(clusters['Prediction'] == behavior) &
                   (clusters['VideoID'] == vid.replace(".mp4", "")) &
                        (clusters['ClipCreated'] == "Yes")]
    clusters['start'] = clusters['t'].apply(lambda x: str(timedelta(seconds=x - t_delta)))

    print(clusters)

