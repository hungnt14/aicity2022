import cv2
import os
from os import listdir
from os.path import isfile, join


def get_timestamp(frame_number,fps):
    timestamp = float(frame_number / fps)
    return timestamp

def write_draft(sub_line, draft_name):
    draft_file = open(draft_name, 'a')
    draft_file.write(sub_line)

def export_submit(threshold, lines, submit_file_path):
    submit = open(submit_file_path, 'w')
    video_ids = get_video_id(lines)
    for video_id in video_ids:
        class_ids = get_class_id(lines, video_id)
        for class_id in class_ids:
            if count_class(class_id, video_id, lines) >= threshold:
                submit.write(get_first_line(class_id, video_id, lines))

# get_video_id
def get_video_id(lines):
    video_id = []
    for line in lines:
        if line.split(' ')[0] not in video_id:
            video_id.append(line.split(' ')[0])
    return video_id

# get first line of each class
def get_first_line(class_id, video_id, lines):
    for line in lines:
        if video_id == line.split(' ')[0] and class_id == line.split(' ')[1]:
            return line

# get all class_id not duplicate
def get_class_id(lines, video_id):
    class_id = []
    for line in lines:
        if video_id == line.split(' ')[0]:
            if line.split(' ')[1] not in class_id:
                class_id.append(line.split(' ')[1])
    return class_id

# count number appearance of each class
def count_class(class_id, video_id, lines):
    count = 0
    for line in lines:
        if video_id == line.split(' ')[0] and class_id == line.split(' ')[1]:
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