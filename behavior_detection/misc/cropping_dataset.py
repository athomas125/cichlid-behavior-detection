import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import affine_transform
import argparse

def get_rotation_matrix(angle):
    angle = np.deg2rad(angle)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])

def process_files(folder_path, angle, x1, y1, x2, y2, output_folder=None, debug_w_dots=False):
    csv_path = None
    h5_path = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            csv_path = os.path.join(folder_path, file_name)
        elif file_name.endswith('.h5'):
            h5_path = os.path.join(folder_path, file_name)

    if not csv_path or not h5_path:
        raise FileNotFoundError("CSV or HDF5 file not found in the specified folder.")

    csv_data = pd.read_csv(csv_path)
    hdf_data = pd.read_hdf(h5_path)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.png'):
            file_path = os.path.join(folder_path, file_name)
            img_id = file_name.split('/')[-1]
            csv_row = [x.split('/')[-1] == img_id for x in csv_data['scorer']].index(True)
            dot_list = []
            with Image.open(file_path) as img:
                center = img.width/2, img.height/2
                transform_matrix = get_rotation_matrix(-angle)
                
                for i in range(1, len(csv_data.columns), 2):
                    x_val = float(csv_data.at[csv_row, csv_data.columns[i]])
                    y_val = float(csv_data.at[csv_row, csv_data.columns[i+1]])
                    if np.isnan(x_val) or np.isnan(y_val):
                        continue
                    rotated_point = np.dot(transform_matrix, [x_val-center[0], y_val-center[1]]) + center
                    if (rotated_point[0] < x1 or rotated_point[1] < y1 or rotated_point[0] > x2 or rotated_point[1] > y2):
                        csv_data.at[csv_row, csv_data.columns[i]] = np.nan
                        csv_data.at[csv_row, csv_data.columns[i+1]] = np.nan
                    else:
                        csv_data.at[csv_row, csv_data.columns[i]] = rotated_point[0] - x1
                        csv_data.at[csv_row, csv_data.columns[i+1]] = rotated_point[1] - y1
                        dot_list.append((csv_data.at[csv_row, csv_data.columns[i]], csv_data.at[csv_row, csv_data.columns[i+1]]))

                rotated_image = img.rotate(angle, center=center, expand=False)
                cropped_img = rotated_image.crop((x1, y1, x2, y2))
                if output_folder:
                    os.makedirs(output_folder, exist_ok=True)
                    cropped_img.save(os.path.join(output_folder, file_name))
                else:
                    cropped_img.save(file_path)

                if debug_w_dots:
                    draw = ImageDraw.Draw(cropped_img)
                    dot_size = 5
                    dot_color = (255, 0, 0)

                    for dot in dot_list:
                        x, y = dot
                        draw.ellipse((x - dot_size, y - dot_size, x + dot_size, y + dot_size), fill=dot_color)

                    output_dot_name = file_name.split('.')[0] + '_dots.png'
                    if output_folder:
                        os.makedirs(output_folder, exist_ok=True)
                        cropped_img.save(os.path.join(output_folder, output_dot_name))
                    else:
                        cropped_img.save(output_dot_name)

    if output_folder:
        csv_data.to_csv(os.path.join(output_folder, os.path.basename(csv_path)), index=False)
    else:
        csv_data.to_csv(csv_path, index=False)

    for i in range(0, len(hdf_data.keys()), 2):
        x_key = hdf_data.keys()[i]
        y_key = hdf_data.keys()[i+1]
        x_data = hdf_data[x_key]
        y_data = hdf_data[y_key]

        transform_matrix = get_rotation_matrix(angle)
        for j in range(len(x_data)):
            x_val = x_data.iloc[j]
            y_val = y_data.iloc[j]
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            rotated_point = np.dot(transform_matrix, [x_val-center[0], y_val-center[1]]) + center
            if (rotated_point[0] < x1 or rotated_point[1] < y1 or rotated_point[0] > x2 or rotated_point[1] > y2):
                x_data.iloc[j] = np.nan
                y_data.iloc[j] = np.nan
            else:
                x_data.iloc[j] = rotated_point[0] - x1
                y_data.iloc[j] = rotated_point[1] - y1

        hdf_data[x_key] = x_data
        hdf_data[y_key] = y_data

    if output_folder:
        hdf_data.to_hdf(os.path.join(output_folder, os.path.basename(h5_path)), key='data', mode='w')
    else:
        hdf_data.to_hdf(h5_path, key='data', mode='w')

    print("Cropping and updates complete.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Crop images and update CSV and HDF5 files.')
    # parser.add_argument('folder', type=str, help='The folder containing the files.')
    # parser.add_argument('x1', type=int, help='The x-coordinate of the top-left corner of the crop area post rotation.')
    # parser.add_argument('y1', type=int, help='The y-coordinate of the top-left corner of the crop area post rotation.')
    # parser.add_argument('x2', type=int, help='The x-coordinate of the bottom-right corner of the crop area post rotation.')
    # parser.add_argument('y2', type=int, help='The y-coordinate of the bottom-right corner of the crop area post rotation.')
    # parser.add_argument('angle', type=float, help='The angle of rotation for the crop area.')
    # parser.add_argument('--output_folder', '-o', type=str, help='Optional output folder to save the modified files.')
    # parser.add_argument('--dot_debug', '-d', type=bool, help='Optional debugging mode to output images with dots on them to verify outputs')

    # args = parser.parse_args()

    # vertices = [(args.x1, args.y1), (args.x2, args.y1), (args.x2, args.y2), (args.x1, args.y2)]
    # process_files(args.folder, vertices, args.angle, args.output_folder, args.dot_debug)

    process_files("clip_5545_5585/", 262, 148, 1194, 842, 3.14, 'rotated', True)
