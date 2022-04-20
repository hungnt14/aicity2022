import os
import argparse
from PIL import Image
from natsort import natsorted
import glob
import time
import cv2

import torch
import torchvision.transforms as T
from general import get_timestamp, export_submit
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="convnext_large")
    parser.add_argument("--model_checkpoint_path", default="")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--videos_input_path", default="/home/data/input/")
    parser.add_argument("--YOLO_folder_path", default="/home/code/yolov5/runs/detect/")
    parser.add_argument("--candidates_file_path", default="/home/data/output/candidates.txt")
    parser.add_argument("--white_tray_infos", default="0.255208333 0.22962963 0.719791667 0.840740741")
    return parser.parse_args()

NORMALIZE_MEAN = IMAGENET_DEFAULT_MEAN
NORMALIZE_STD = IMAGENET_DEFAULT_STD
SIZE = 256

# const 
VIDEO_NAME_PREFIX = ""
VIDEO_FPS = []

# Here we resize smaller edge to 256, no center cropping
transforms = [
              T.Resize(SIZE, interpolation=T.InterpolationMode.BICUBIC),
              T.ToTensor(),
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
              ]

transforms = T.Compose(transforms)

def load_model(model_name, checkpoint, num_gpus):
    # create model
    model = create_model(
            model_name,
            num_classes=116,
            in_chans=3,
            checkpoint_path=checkpoint)
    
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpus))).cuda()
    else:
        model = model.cuda()
    model.eval()
    return model

def load_image(image_path, device):
    img = Image.open(image_path)
    rgb_im = img.convert('RGB')
    img_tensor = transforms(rgb_im).unsqueeze(0).to(device)

    return img_tensor

def predict(image_path, model_name, checkpoint):
    model = load_model(model_name, checkpoint)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_tensor = load_image(image_path, device)

    labels = model(img_tensor)
    topk = labels.topk(5)

    return topk

def get_latest_YOLO_execution_folder(YOLO_folder_path):
    return list(reversed(natsorted(glob.glob(YOLO_folder_path + "*"))))[0]

def load_video_info(videos_input_path):
    global VIDEO_NAME_PREFIX, VIDEO_FPS
    video_paths = glob.glob(videos_input_path + "*")
    VIDEO_FPS = [60 for i in range(len(video_paths))]

    for video_path in video_paths:
        VIDEO_NAME_PREFIX, video_id = os.path.basename(video_path).split(".")[0].split("_")
        # get video fps
        videoCapture = cv2.VideoCapture(video_path)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)

        # 0-index list
        VIDEO_FPS[int(video_id) - 1] = fps

def get_yolo_frame_info(video_id, frame_id, object_id_in_img, latest_YOLO_folder_path):

    label_name = VIDEO_NAME_PREFIX + "_" + str(video_id) + "_" + str(frame_id) + ".txt"
    label_folder = latest_YOLO_folder_path + "/labels/"
    with open(label_folder + label_name, 'r') as f:
        label_infos = f.readlines()
     
    return label_infos[object_id_in_img].replace("\n", "").split(" ")

def process(videos_input_path, model_name, model_checkpoint_path, num_gpus, YOLO_folder_path, candidates_file_path, white_tray_infos):
    load_video_info(videos_input_path)
    
    model = load_model(model_name, model_checkpoint_path, num_gpus)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    latest_YOLO_folder_path = get_latest_YOLO_execution_folder(YOLO_folder_path)
    cropped_folder_path = latest_YOLO_folder_path + "/crops/"
    image_paths = natsorted(glob.glob(cropped_folder_path + "*"))

    candidates = []
    for image_path in image_paths:

        image_name = os.path.basename(image_path).split(".")[0]
        image_name_split = image_name.split("_")

        # fetch object infos
        video_id = image_name_split[1]
        frame_id = int(image_name_split[2])

        timestamp = frame_id / VIDEO_FPS[int(video_id) - 1]

        if(len(image_name_split) == 4):
            object_id_in_img = int(image_name_split[3]) - 1
        else:
            object_id_in_img = 0
        
        yolo_class_id, x, y, w, h = get_yolo_frame_info(video_id, frame_id, object_id_in_img, latest_YOLO_folder_path)
        x, y, w, h = float(x), float(y), float(w), float(h)

        # get white tray infos
        wt_xmin, wt_ymin, wt_xmax, wt_ymax = [float(x) for x in white_tray_infos.split(" ")]
        
        
        if (x > wt_xmax or x < wt_xmin or y > wt_ymax or y < wt_ymin):
            # if coordinate of center of current candidate is not valid, then skip
            continue
        
        # neu dien tich box lon hon dien tich khay thi vut
        #print(x,y, wt_xmin, wt_ymin, wt_xmax, wt_ymax)
        # start classifying
        img_tensor = load_image(image_path, device)

        pred = model(img_tensor)
        
        convnext_class_id = int(pred.topk(1).indices.item()) + 1
        values, indices = pred.topk(1)
        convnext_conf = values.item()

        candidate = [video_id, frame_id, timestamp, convnext_class_id, yolo_class_id, x, y, w, h, convnext_conf]
        candidates.append(candidate)
    
    export_content = ""
    for candidate in candidates:
        video_id, frame_id, timestamp, convnext_class_id, yolo_class_id, x, y, w, h, convnext_conf = candidate
        export_content += "{} {} {} {} {} {} {} {} {} {}\n".format(video_id, frame_id, timestamp, convnext_class_id, yolo_class_id, x, y, w, h, convnext_conf)
    with open(candidates_file_path, "w") as f:
        f.write(export_content)

if __name__ == "__main__":
    args = get_parser()
    
    print("=== Start classify")
    start_time = time.time()
    process(args.videos_input_path,
            args.model_name, 
            args.model_checkpoint_path, 
            args.num_gpus,
            args.YOLO_folder_path, 
            args.candidates_file_path,
            args.white_tray_infos)
    print("=== Done classify. Time elapsed:", str(time.time() - start_time))
