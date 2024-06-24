import sys
import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the repository root directory to sys.path
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    
from behavior_detection.misc.cropping_dataset import crop_datasets

# crop_datasets(folder, angle, x1, x2, x3, x4, output_folder, debug_w_dots)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc23_1_Tk33_021220_0004_vid', 0, 87, 89, 1112, 914, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc23_8_Tk33_031720_0001_vid', 0, 89, 83, 1104 ,911, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc26_2_Tk63_022520_0001_vid', -0.92, 225, 243, 1071, 926, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc28_1_Tk3_022520_0004_vid', 5.08, 324, 8, 1184, 810, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc29_3_Tk9_030320_0001_vid', -1.17, 336, 128, 1185, 825, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc32_5_Tk65_030920_0002_vid', -2.62, 276, 138, 1124, 818, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc34_3_Tk43_030320_0001_vid', 0, 60, 29, 1116, 873, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc35_11_Tk61_051220_0002_vid', 0, 285, 117, 1131, 791, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc36_2_Tk3_030320_0001_vid', 5.12, 345, 2, 1187, 807, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc37_2_Tk17_030320_0001_vid', 0, 257, 35, 948, 887, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc40_2_Tk3_030920_0002_vid', 5.81, 250, 0, 1181, 806, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc41_2_Tk9_030920_0001_vid', -1.5, 332, 126, 1185, 825, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc43_11_Tk41_060220_0001_vid', 1.66, 81, 0, 1139, 783, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc44_7_Tk65_050720_0002_vid', -2.43, 261, 144, 1121, 957, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc45_7_Tk47_050720_0002_vid', 1.89, 123, 129, 978, 803, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc55_2_Tk47_051220_0002_vid', 1.84, 140, 6, 983, 816, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc56_2_Tk65_051220_0002_vid', -2.02, 273, 149, 1115, 932, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc59_4_Tk61_060220_0001_vid', 0, 287, 117, 1134, 960, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc62_3_Tk65_060220_0001_vid', -1.88, 272, 153, 1115, 833, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc63_1_Tk9_060220_0001_vid', -1.53, 341, 125, 1193, 828, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc64_1_Tk51_060220_0002_vid', 0, 251, 129, 1089, 807, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc65_4_Tk9_072920_0002_vid', -1.75, 335, 116, 1200, 836, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc76_3_Tk47_072920_0002_vid', 1.64, 134, 126, 877, 800, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc80_1_Tk41_072920_0002_vid', 1.41, 81, 0, 1136, 789, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc81_1_Tk51_072920_0002_vid', 0, 249, 0, 1094, 806, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc82_b2_Tk63_073020_0002_vid', 0, 222, 233, 1061, 930, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc86_b1_Tk47_073020_0002_vid', 2.07, 149, 5, 974, 806, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc90_b1_Tk3_081120_0001_vid', 6.33, 346, 2, 1184, 805, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc91_b1_Tk9_081120_0001_vid', -1.41, 335, 9, 1190, 824, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc94_b1_Tk31_081120_0001_vid', 0, 258, 35, 951, 884, debug_w_dots=True)
crop_datasets('/data/home/athomas314/dlc_model-student-2023-07-26_cropped/labeled-data/MC_singlenuc96_b1_Tk41_081120_0001_vid', 1.76, 80, 0, 1137, 792, debug_w_dots=True)