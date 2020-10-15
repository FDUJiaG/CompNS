#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import moviepy.editor as mpe
import cv2


def get_time(date=False, utc=False, msl=3):
    if date:
        time_fmt = "%Y-%m-%d %H:%M:%S.%f"
    else:
        time_fmt = "%H:%M:%S.%f"

    if utc:
        return datetime.utcnow().strftime(time_fmt)[:(msl-6)]
    else:
        return datetime.now().strftime(time_fmt)[:(msl-6)]


def print_info(status="I"):
    return "\033[0;33;1m[{} {}]\033[0m".format(status, get_time())


def find_file_list(path, file_type):
    return glob.glob(os.path.join(path, file_type))


def op_dict_judge(op_dict, op_item, f_style_dict):
    if len(op_dict[op_item]) == 0:
        print(print_info("E"), end=" ")
        print("No documents meeting the '{}' format. Please Check Again!".format(
            f_style_dict[op_item]
        ))
        return False

    elif len(op_dict[op_item]) > 1:
        print(print_info("W"), end=" ")
        print("We only support one '{}' file in the root path now. Please Check Again!".format(
            f_style_dict[op_item]
        ))
        return False

    else:
        return True


def get_root_path(r_path="."):
    root_path = os.path.abspath(r_path)
    print(print_info(), end=" ")
    print("The root path is: '{}'.".format(root_path))

    return root_path


def get_op_dict(root_path, op_list, f_style_dict):
    op_dict = dict()
    for op_item in op_list:
        op_dict[op_item] = find_file_list(root_path, "*" + f_style_dict[op_item])

    return op_dict


def get_output_dirs(r_path, o_dir="outputs", f_dir="figs", op_f_dir="op_figs"):
    # set output dir
    output_dir = os.path.join(r_path, o_dir)
    if not os.path.exists(output_dir):
        # If outputs directory does not exist, create a directory
        os.makedirs(output_dir)
        print(print_info(), end=" ")
        print("The output directory '{}' is successfully created!".format(output_dir))

    # set figs sequence dir
    figs_dir = os.path.join(output_dir, f_dir)
    # If figs directory does exist, delete it before create a new one.
    if os.path.exists(figs_dir):
        shutil.rmtree(figs_dir)
        print(print_info("W"), end=" ")
        print("The old figs directory: '{}' is removed!".format(figs_dir))

    os.makedirs(figs_dir)
    print(print_info(), end=" ")
    print("The figs directory '{}' is successfully created!".format(figs_dir))

    # set operating figs sequence dir
    op_figs_dir = os.path.join(output_dir, op_f_dir)
    if os.path.exists(op_figs_dir):
        shutil.rmtree(op_figs_dir)
        print(print_info("W"), end=" ")
        print("The old operating figs directory: '{}' is removed!".format(op_figs_dir))

    os.makedirs(op_figs_dir)
    print(print_info(), end=" ")
    print("The operating figs directory '{}' is successfully created!".format(op_figs_dir))

    return output_dir, figs_dir, op_figs_dir


def loc_read(loc_file, header_list=[0, 1, 2]):
    try:
        print(print_info(), end=" ")
        print("Loading the location dataframe...")
        df = pd.read_csv(loc_file, header=header_list, index_col=0)
        print(print_info(), end=" ")
        print("The location dataframe from '{}' is loaded!".format(loc_file))
        print(print_info(), end=" ")
        print("The information of location file is:")
        print(df.info)
        return df
    except:
        print(print_info("E"), end=" ")
        print("Could not loading the '{}'. Please check again!".format(loc_file))
        return False


def get_columns(df, is_print=True):
    columns_names = list(df.columns.names)
    columns_dicts = dict()
    for idx in range(len(columns_names)):
        columns_dicts[columns_names[idx]] = [item[idx] for item in df.columns]

    if is_print:
        print(print_info(), end=" ")
        print("The column name struct is:")

        print_col_dict = columns_dicts.copy()

        for key, value in print_col_dict.items():
            print_col_dict[key] = list(set(value))
        print(json.dumps(print_col_dict, indent=4))

    return columns_dicts


def get_loc_name_idx(columns_dicts, axis_n, part_n):
    location_names = list(zip(
        [val for val in part_n for i in range(2)],
        axis_n * len(part_n)
    ))

    location_index = [
        list(zip(
            columns_dicts["bodyparts"],
            columns_dicts["coords"]
        )).index(item) for item in location_names
    ]

    loc_col_name = [
        '_'.join(location_names[idx]) for idx in range(len(location_names))
    ]

    return loc_col_name, location_index


def make_loc_df(df, loc_col_name, loc_index):
    loc_df = pd.DataFrame(columns=loc_col_name)

    for idx in range(len(loc_col_name)):
        loc_df[loc_col_name[idx]] = df.iloc[:, loc_index[idx]].map(lambda x: int(round(x, 0)))

    return loc_df


def get_loc_df(file_path, axis_n, part_n):
    df = loc_read(file_path)

    if type(df) is bool:
        return False

    columns_dicts = get_columns(df)
    loc_col_name, loc_index = get_loc_name_idx(columns_dicts, axis_n, part_n)

    return make_loc_df(df, loc_col_name, loc_index)


def get_video(file_path):
    try:
        print(print_info(), end=" ")
        print("Loading the video file...")
        video = mpe.VideoFileClip(file_path, audio=False)
        print(print_info(), end=" ")
        print("The video file from '{}' is loaded!".format(file_path))
    except:
        print(print_info("E"), end=" ")
        print("Could not loading the '{}'. Please check again!".format(file_path))
        return False
    else:
        return video


def get_video_info(video):
    v_size, v_fps = video.size, int(video.fps)
    v_total_fps = int(video.duration * video.fps)
    print(print_info(), end=" ")
    print("The video size is {}, fps is {}.".format(v_size, v_fps))

    return v_size, v_fps, v_total_fps


def video_to_figs_sequence(video, figs_dir, f_style_dict):
    # cut the video to images sequence
    try:
        print(print_info(), end=" ")
        print("Cutting video to images sequence...")
        video.write_images_sequence(
            os.path.join(figs_dir, "frame%06d" + f_style_dict["figure"]),
            verbose=False
        )
        return True
    except:
        print(print_info("E"), end=" ")
        print("Could not cut video to images sequence. Please Check Again!")
        return False


def get_pixel_value(frame, xy_list):
    pixel_value = [
        frame[item[0], item[1], :].tolist() for item in xy_list
    ]

    return pixel_value


def rgb_to_temp(rgb_items, change_mat=[0.2989, 0.5870, 0.1140]):
    gray = np.dot(rgb_items, change_mat)
    temperature = round(1 - gray / 255, 3)
    return temperature


def frame_operator(frame, xy_list, pixel_value, part_name, v_size, reduce_rate=100, linewidth=2):
    frame = cv2.UMat(frame)
    frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rate_step = 0.06
    rect_rate = [
        0.68,
        0.02,
        0.98,
        0.01 + rate_step * len(part_name)
    ]

    rect_loc = [
        (int(v_size[0] * rect_rate[0]), int(v_size[1] * rect_rate[1])),
        (int(v_size[0] * rect_rate[2]), int(v_size[1] * rect_rate[3]))
    ]
    cv2.rectangle(frame_show, rect_loc[0], rect_loc[1], [0, 0, 255], linewidth)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for idx, item in enumerate(pixel_value):
        cv2.putText(
            frame_show, part_name[idx],
            (
                int(v_size[0] * (rect_rate[0] + 0.01)),
                int(v_size[1] * (rate_step * (idx + 0.5) + rect_rate[1]))
            ),
            font, 0.5, [0, 0, 0], 2
        )
        cv2.putText(
            frame_show, str(rgb_to_temp(item)),
            (
                int(v_size[0] * (rect_rate[0] * 0.4 + rect_rate[2] * 0.6)),
                int(v_size[1] * (rate_step * (idx + 0.5) + rect_rate[1]))
            ),
            font, 0.5, [0, 0, 0], 2
        )

    # change pixel size to plot size
    f_size = [item / reduce_rate for item in v_size]

    for idx, item in enumerate(xy_list):
        cv2.circle(frame_show, tuple(item), int(min(f_size) * linewidth - 1), [0, 0, 0], linewidth)
        cv2.putText(frame_show, part_name[idx], tuple(item), font, 0.5, [0, 0, 0], linewidth)

    return frame_show


def time_format(m):
    if len(str(m)) == 1:
        return "0{}".format(m)
    else:
        return "{}".format(m)


def op_fig_name(f_index, pre_name="labeled_frame", frame_style=".jpeg", op_fps=20):
    m_per_fps = op_fps * 60

    f_fps = f_index % op_fps
    f_sec = f_index // op_fps
    f_min = f_index // m_per_fps
    f_sec = f_sec % 60

    frame_label = [
        pre_name,
        time_format(f_min),
        time_format(f_sec),
        time_format(f_fps)
    ]

    return "_".join(frame_label) + frame_style


def get_all_frame(
    input_list, video, v_total_fps, op_df, output_dir, v_size, part_name,
    op_fps=20, v_vs_loc=True, reduce_rate=100
):

    input_len = len(input_list)

    if input_len == 0:
        min_v_fps = v_total_fps

    else:
        if input_len - v_total_fps > 1:
            print(print_info("E"), end=" ")
            print("The figures number {} is not right. Please Check Again!".format(input_len))
            return False

        min_v_fps = min(input_len, v_total_fps)

    df_len = op_df.shape[0]

    if v_vs_loc:
        if abs(min_v_fps - df_len) > op_fps / 10:
            print(print_info("E"), end=" ")
            print("The video length {} and marker length {} is not equal. Please Check Again!".format(
                input_len, df_len
            ))
            return False

    if min_v_fps != df_len:
        op_len = min(min_v_fps, df_len)
        print(print_info("W"), end=" ")
        print(
            "The video length {} and location length {} is not equal, " \
            "it may cause {:.2f}(s) drift.".format(
                min_v_fps, df_len, abs(min_v_fps - df_len) / op_fps
            ))

        print(print_info("W"), end=" ")
        print("Change the operate length to {}.".format(op_len))
    else:
        op_len = min_v_fps

    print(print_info(), end=" ")
    print("Operateing the frames sequence...")

    for idx in range(op_len):
        # frame = cv2.imread(input_list[idx])
        frame = video.get_frame(idx)
        xy_list = op_df.values[idx].reshape(-1, 2)
        pixel_value = get_pixel_value(frame, xy_list)
        op_frame = frame_operator(frame, xy_list, pixel_value, part_name, v_size, reduce_rate)
        cv2.imwrite(
            os.path.join(output_dir, op_fig_name(idx)),
            op_frame
        )

    print(print_info(), end=" ")
    print("The operated frames sequence is saved to '{}'.".format(output_dir))

    return True


def labeled_video_output(op_figs_dir, v_fps, output_dir, output_file_name="output"):
    try:
        print(print_info(), end=" ")
        print("Getting the operated frames sequence...")
        clip = mpe.ImageSequenceClip(op_figs_dir, fps=v_fps)
        print(print_info(), end=" ")
        print("Operated frames sequence getted!")
    except:
        print(print_info("E"), end=" ")
        print("Cloud not get the operated frames sequence. Please check again!")
        return False
    try:
        print(print_info(), end=" ")
        print("Compositing video file from operated frames sequence...")
        clip.write_videofile(os.path.join(output_dir, output_file_name + ".mp4"), fps=v_fps, audio=False)
        print(print_info(), end=" ")
        print("Video file compositted at '{}'!".format(os.path.join(output_dir, output_file_name + ".mp4")))
    except:
        print(print_info("E"), end=" ")
        print("Cloud not compositing video file. Please check again!")
        return False
    return True


def temperature_operator(f_style_dict, op_list, axis_name, part_name, is_out_ori_figs=False):
    print(print_info(), end=" ")
    print("OpenCV Version is:", cv2.__version__)

    root_path = get_root_path()
    op_dict = get_op_dict(root_path, op_list, f_style_dict)

    for op_item in op_list:
        if not op_dict_judge(op_dict, op_item, f_style_dict):
            return False

    output_dir, figs_dir, op_figs_dir = get_output_dirs(root_path)

    # check location file
    loc_df = get_loc_df(
        op_dict[op_list[1]][0],
        axis_name, part_name
    )

    if type(loc_df) is bool:
        print(print_info("E"), end=" ")
        print("There is something wrong in the location information operating. Please check again!")
        return False

    # check video file
    video = get_video(op_dict[op_list[0]][0])

    if not video:
        return False
    else:
        v_size, v_fps, v_total_fps = get_video_info(video)

    if is_out_ori_figs:
        if not video_to_figs_sequence(video, figs_dir, f_style_dict):
            return False

    # output
    figs_list = sorted(find_file_list(figs_dir, "*" + f_style_dict["figure"]))
    if not get_all_frame(
            figs_list, video, v_total_fps, loc_df, op_figs_dir, v_size, part_name, v_fps, True
    ):
        return False

    return labeled_video_output(op_figs_dir, v_fps, output_dir)


def main():
    msl = 3
    start_time = datetime.now()

    f_style_dict = {
        "video": ".avi",
        "location": ".csv",
        "figure": ".jpeg"
    }

    op_list = ["video", "location"]

    axis_name = ["x", "y"]
    part_name = [
        "bodyW",
        "tailbaseW"
    ]

    if temperature_operator(f_style_dict, op_list, axis_name, part_name):
        end_time = datetime.now()
        print(print_info(), end=" ")
        print("Total time used: {}".format(str(end_time - start_time)[:(msl-6)]))
        print(print_info("S"), end=" ")
        print("All operator is done!")

        return True

    return False


if __name__ == '__main__':
    judge = main()
