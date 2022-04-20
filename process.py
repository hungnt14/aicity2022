# utility libraries
import argparse
from calendar import c
import os
import glob
import time

from natsort import natsorted
import cv2
import numpy as np

# special libraries
from utils import *

# args parser
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", default="")
    parser.add_argument("--input_path", default="")
    parser.add_argument("--input_video_paths", default="")
    parser.add_argument("--output_path", default="")
    return parser.parse_args()

def get_latest_execution_folder(folder_paths):
    return list(reversed(natsorted(folder_paths)))[0]

def get_xy_minmax_box_coordinates(arr):
    x, y, w, h = arr
    # video constant
    frame_width = 1920
    frame_height = 1080
    
    # get xmin, ymin, xmax, ymax.
    x *= frame_width
    y *= frame_height
    w *= frame_width
    h *= frame_height

    return (
        (x - w / 2) / frame_width,
        (y - h / 2) / frame_height,
        (x + w / 2) / frame_width,
        (y + h / 2) / frame_height
    )

def getIOU(boxA, boxB):
    boxA = get_xy_minmax_box_coordinates(boxA)
    boxB = get_xy_minmax_box_coordinates(boxB)
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
    return iou

def pick_class_id(candiate_file, video_id):
    # weight of number of each class id
    w_0 = 50
    # weight of average confidence
    w_1 = 1
    # weight of overlap
    w_2 = 0
    # threshold
    threshold = 6
    lines = open(candiate_file).readlines()

    results = []
    class_ids = get_class_id(lines, video_id)
    for class_id in class_ids:
        n_class = count_class(class_id, video_id, lines)
        conf = cal_average_conf(class_id, video_id, lines)
        n_overlap = cal_overlap_model(class_id, video_id, lines)
        score = cal_score(w_0, w_1, w_2, class_id, video_id, lines)
        if score > threshold:
            results.append(class_id)
    
    return results

def filter_candidates_list(input_path, output_path):
    
    print("=== Start filter_candidates_list")
    start_time = time.time()

    # constant:
    coordinate_eps = 0.1 # (in ratio) distance between current center and last center
    min_num_of_frames = 17 # (frames) minimum frames to be accepted for exporting result
    f_eps = 5 # (frames) distance from current frame to the last recorded frames

    # bot giat bot giat moi
    # goi do an
    # candidates list format:
    # <video_id: int> <frame_id: int> <class_id: int> <x y w h: float>
    with open(input_path, "r") as f:
        lines = f.readlines()

    videos = {}
    for line in lines:
        line = line.replace("\n", "").split(" ")
        video_id = line[0]
        if (video_id not in videos):
            videos[video_id] = []
        videos[video_id].append([float(x) if i != 0 and i != 3 and i != 4 else x for i, x in enumerate(line)])

    results = []
    for video_id in videos:
        object_list = []
        for candidate in videos[video_id]:
            _, frame_id, timestamp, class_id, yolo_class_id, x, y, w, h, convnext_conf = candidate
            matched = False
            for i, object in enumerate(object_list):
                if (abs(frame_id - object["last_frame"]) <= f_eps and \
                    object["class_id"] == class_id and \
                    getIOU(object["last_center_xy"] + object["last_center_wh"], [x, y, w, h]) > 0.7):
                    # valid
                    matched = True
                    object_list[i]["last_frame"] = frame_id
                    object_list[i]["last_center_xy"] = (x, y)
                    object_list[i]["last_center_wh"] = (w, h)
                    object_list[i]["count_frame"] += 1
                    object_list[i]["sum_timestamp"] += timestamp
                    break

            if (not matched):
                # if not matched with any exist object, create a new one
                object_list.append({
                    "class_id": class_id,
                    "first_frame": frame_id,
                    "last_frame": frame_id,
                    "last_center_xy": (x, y),
                    "last_center_wh": (w, h),
                    "count_frame": 1,
                    "sum_timestamp": timestamp
                })
        
        # get wanted class_id and filter from these blocks
        wanted_class_ids = pick_class_id(input_path, video_id)

        #print(video_id, wanted_class_ids)
        for wanted_class_id in wanted_class_ids:
            sum_count_frame = 0
            sum_timestamp = 0
            max_count_frame = 0
            for object_id, object in enumerate(object_list):
                if (object["class_id"] == wanted_class_id):
                    if (object["count_frame"] > max_count_frame):
                        max_count_frame = object["count_frame"]
                        
                        sum_count_frame = object["count_frame"]
                        sum_timestamp = object["sum_timestamp"]
                    
            timestamp = sum_timestamp / sum_count_frame
            results.append([video_id, timestamp, wanted_class_id])
        print("Done filtering", video_id)
    
    sorted(results, key=lambda x: x[1])
    export_content = ""
    for i, result in enumerate(results):
        video_id, timestamp, class_id = result
        if (i > 0 and int(timestamp) == int(results[i - 1][1]) and class_id == results[i - 1][2]):
            continue
        export_content += "{} {} {}\n".format(video_id, class_id, int(timestamp))
    
    with open(output_path, "w") as f:
        f.write(export_content)

    print("=== Done filter_candidates_list. Time elapsed:", str(time.time() - start_time))

if __name__ == "__main__":
    args = get_parser()
    if (args.module == "" or args.input_path  == "" or args.output_path == ""):
        print("Required arguments")
        exit()
    else:
        if (args.module == "extract_white_tray_infos"):
             extract_white_tray_infos(args.input_path, args.input_video_paths, args.output_path)
        if (args.module == "crop_white_tray_in_videos"):
            crop_white_tray_in_videos(args.input_path, args.input_video_paths, args.output_path)
        if (args.module == "filter_candidates_list"):
            filter_candidates_list(args.input_path, args.output_path)