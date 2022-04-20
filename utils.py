import cv2
import os
from os import listdir
from os.path import isfile, join

# video_id, frame_id, timestamp, convnext_class_id, yolo_class_id, x, y, w, h, convnext_conf
def get_timestamp(frame_number,fps):
    timestamp = float(frame_number / fps)
    return timestamp

# get_video_id
def get_video_id(lines):
    video_id = []
    for line in lines:
        if line.split(' ')[0] not in video_id:
            video_id.append(line.split(' ')[0])
    return video_id

def export_submit(submit_name, class_id, video_id, lines):
    submit = open(submit_name, 'a')
    submit.write(get_first_line(class_id, video_id, lines))

# get first line of each class
def get_first_line(class_id, video_id, lines):
    for line in lines:
        if video_id == line.split(' ')[0] and class_id == line.split(' ')[3]:
            split_lines = line.split(' ')
            return str(split_lines[0]) + " " + str(split_lines[3]) + " " + str(int(float(split_lines[2]) + 0.5)) + "\n"

# get all class_id not duplicate
def get_class_id(lines, video_id):
    class_id = []
    for line in lines:
        if video_id == line.split(' ')[0]:
            if line.split(' ')[3] not in class_id:
                class_id.append(line.split(' ')[3])
    return class_id


def cal_score(w_0, w_1, w_2, class_id, video_id, lines):
    n_class = count_class(class_id, video_id, lines)
    n_vid = count_video_lines(video_id, lines)
    conf = cal_average_conf(class_id, video_id, lines)
    n_overlap = cal_overlap_model(class_id, video_id, lines)
    score = w_0 * (n_class / n_vid) + w_1 * conf + w_2 * (n_overlap / n_vid)
    return score

def cal_average_conf(class_id, video_id, lines):
    n = count_class(class_id, video_id, lines)
    sum = 0
    for line in lines:
        if video_id == line.split(' ')[0] and class_id == line.split(' ')[3]:
            conf = float(line.split(' ')[9])
            sum += conf
    return sum / n
            
def cal_overlap_model(class_id, video_id, lines):
    n_overlap = 0
    for line in lines:
        if video_id == line.split(' ')[0] and class_id == line.split(' ')[3]:
            if(int(line.split(' ')[3]) == int(line.split(' ')[4])+1):
                n_overlap += 1
    return n_overlap

# count number appearance of each class
def count_class(class_id, video_id, lines):
    count = 0
    for line in lines:
        if video_id == line.split(' ')[0] and class_id == line.split(' ')[3]:
            count += 1
    return count

# count number of lines of each video
def count_video_lines(video_id, lines):
    count = 0
    for line in lines:
        if video_id == line.split(' ')[0]:
            count += 1
    return count

def crop(image_path, crop_path, x1, y1, x2, y2):
    image = cv2.imread(image_path)
    crop_img = image[y1:y2, x1:x2]
    cv2.imwrite(crop_path, crop_img)

# get all files in a folder and subfolder
def get_files(folder):
    files = [f for f in listdir(folder) if isfile(join(folder, f))]
    return files

