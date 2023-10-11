#!/usr/bin/env python
"""
    This script extracts potential 'bower circling' clips from a video
    @author: mikster36
    @date: 10/2/23
"""
import itertools
import os
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import pandas as pd
import numpy as np

filepath_h5 = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/"
               r"bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el.h5")
filepath_pickle = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                   r"/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_assemblies.pickle")
video = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
         r"/bower_circling/bower_circlingDLC_dlcrnetms5_dlc_modelJul26shuffle4_100000_el_filtered_id_labeled.mp4")
out = (r"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520/bower_circling"
       r"/labeled-frames")
matplotlib.use("TKAgg")


@dataclass
class Vel:
    direction: np.ndarray
    magnitude: float


@dataclass
class Fish:
    position: Any
    vel: list[Vel]


def read_video(video: str, output: str):
    import cv2
    vid = cv2.VideoCapture(video)
    success, image = vid.read()
    count = 0
    while success:
        cv2.imwrite(f"{os.path.join(output, f'frame{count}.png')}", image)
        success, image = vid.read()
        print('Read a new frame: ', success)
        count += 1


def show_nframes(frames: str, n: int):
    import cv2
    for i in range(n):
        image = cv2.imread(f"{os.path.join(frames, f'frame{i}.png')}")
        window_name = f'frame{i}'
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()


def plot_velocities(frame: str, fishes: list[Fish], destfolder: str, show=False):
    img = Image.open(frame)
    img_data = np.flipud(np.array(img))
    fig, ax = plt.subplots()
    ax.imshow(img_data, origin='upper')
    color = itertools.cycle(('red', 'blue'))

    for i, fish in enumerate(fishes):
        # plot each body part's velocity
        fishcolor = next(color)
        for velocity, position in zip(fish.vel, fish.position):
            x, y = position
            y = img.height - y
            dx, dy = velocity.magnitude * velocity.direction
            dy = -dy
            ax.add_patch(patches.Arrow(x, y, dx=dx, dy=dy, width=5, color='white'))
            ax.plot(x, y, marker='.', color=fishcolor, markersize=1)
        print()

    ax.set_xlim(0, img.width)
    ax.set_ylim(0, img.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(os.path.join(destfolder, f"{os.path.basename(frame).split('.')[0]}-velocities.png"),
                dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    if show:
        plt.imshow()


def get_centroid(xy_coords: list[tuple]):
    x_sum, y_sum = 0, 0
    for i in xy_coords:
        x_sum += i[0]
        y_sum += i[1]
    return np.array([x_sum / len(xy_coords), y_sum / len(xy_coords)])


def get_approximations(frame):
    """
        Approximates each fish in a frame to three clusters: front, middle, tail

        Args:
            frame: list of dicts - a frame with each detection
        Returns:
            a list of dicts where each key is a fish and its value is a matrix of its
            body clusters' positions and likelihoods
    """
    bodies = {}
    for fish, matrix in frame.items():
        # front cluster is the centroid of nose, lefteye, righteye, and spine1
        # middle cluster is the centroid of spine2, spine3, leftfin, and rightfin
        # tail is the backfin
        # these approximations help abstract the fish and its movement
        front_cluster = [(matrix[i][0], matrix[i][1]) for i in range(4)]
        middle_cluster = [(matrix[i][0], matrix[i][1]) for i in range(3, 9) if i != 6]
        front = get_centroid(front_cluster)
        centre = get_centroid(middle_cluster)
        tail = (matrix[6][0], matrix[6][1])
        bodies.update({fish: np.array([front, centre, tail])})
    return bodies


def get_velocities(pi: np.ndarray, pf: np.ndarray, n: int):
    """
    Gets the velocity of a fish in a frame by calculating the velocity of each body mass

    Args:
        pi (tuple [3-tuple of tuples]): the initial position of a cichlid (head, centre, tail)
        pf (tuple [3-tuple of tuples]): the final position of a cichlid (head, centre, tail)
        n (int): the time (in frames) between final and initial position
    Returns:
        tuple (3-tuple of tuples): the velocity for each body mass in the form: (direction, magnitude)
    """
    front_vel = np.array([pf[0][0] - pi[0][0], pf[0][1] - pi[0][1]]) / n
    centre_vel = np.array([pf[1][0] - pi[1][0], pf[1][1] - pi[1][1]]) / n
    tail_vel = np.array([pf[2][0] - pi[2][0], pf[2][1] - pi[2][1]]) / n
    return (Vel(front_vel / np.linalg.norm(front_vel), np.linalg.norm(front_vel)),
            Vel(centre_vel / np.linalg.norm(centre_vel), np.linalg.norm(centre_vel)),
            Vel(tail_vel / np.linalg.norm(tail_vel), np.linalg.norm(tail_vel)))


def df_to_reshaped_list(df: pd.DataFrame):
    """
    By default, the *_el.h5 file is stored as a DataFrame with a shape and organisation
    similar to how the csv file looks, i.e.

    individual  fish1 fish1 fish1       ...
    bodypart    nose  nose  nose        ...
    coords      x     y     likelihood  ...
    frame #
    ...

    This method reshapes that data into a list of dicts, where the key is the fish and
    its value is a matrix of the following shape. The frame number is the index of the dict

                x   y   likelihood
    nose
    lefteye
    ...
    rightfin
    """
    frames = list()
    for i in range(df.shape[0]):
        frame = df.iloc[i].droplevel(0).dropna()
        frame = frame.unstack(level=[0, 2])
        frame: pd.DataFrame = frame.reindex(['nose', 'lefteye', 'righteye',
                                             'spine1', 'spine2', 'spine3',
                                             'backfin', 'leftfin', 'rightfin'])
        framedict = {fish: frame[fish].to_numpy() for fish in frame.columns.get_level_values(0).unique()}
        frames.append(framedict)
    return frames


def same_fish_in_both(d1: dict, d2: dict):
    return bool(set(d1.keys()) & set(d2.keys()))


if __name__ == "__main__":
    tracklets: pd.DataFrame = pd.DataFrame(pd.read_hdf(filepath_h5))
    frames = df_to_reshaped_list(tracklets)
    start_index = 70
    nframes = 200

    frames = [frames[i + start_index] for i in range(nframes)]

    for i in range(1, 2):
        prev_frame, curr_frame = frames[i - 1], frames[i]
        prev_bodies, curr_bodies = get_approximations(prev_frame), get_approximations(curr_frame)
        if not same_fish_in_both(prev_bodies, curr_bodies):
            continue
        # gets the velocities of fish appearing in both frames
        vels = {key : get_velocities(prev_bodies.get(key), curr_bodies.get(key), 1)
                for key in curr_bodies.keys() if key in prev_bodies}
        fishes = {fish : Fish(position=prev_bodies.get(fish), vel=vel) for fish, vel in vels.items() if fish in prev_bodies}
        frame_path = (f"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                      f"/bower_circling/frames/")
        frame_path = os.path.join(frame_path, f"frame{i + start_index - 1}.png")
        dest_folder = (f"/home/bree_student/Downloads/dlc_model-student-2023-07-26/videos/MC_singlenuc26_2_Tk63_022520"
                       f"/bower_circling/velocities/")
        # needs updating
        plot_velocities(frame=frame_path, fishes=fishes, destfolder=dest_folder, show=False)

