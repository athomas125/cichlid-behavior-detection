from behavior_detection.BehavioralClip import BehavioralClip
#from behavior_detection.misc_scripts.analyse_videos import analyse_videos
import behavior_detection.bower_circling as bc

if __name__ == "__main__":
    config_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/config.yaml"
    video_path = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circling.mp4"
    tracklets = r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered.h5"
    behavioralClip = BehavioralClip(video_path, tracklets)
    behavioralClip.calculate_velocities(smooth_factor=7, mask_xy=(172, 199), mask_dimensions=(945, 749))
    bc.track_bower_circling(behavioralClip.velocities)
