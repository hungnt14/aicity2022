# yolov5 product detection
python3 yolov5/custom_detect.py --device 0 --conf-thres 0.4 --save-crop --save-txt \
               --weights /home/weights/product_detection_v11.pt \
               --source /home/data/input/

# convnext classifying
python3 convnext/classify.py --model_name convnext_large \
                            --model_checkpoint_path /home/weights/convnext-large-v1.pth.tar \
                            --num_gpus 1 \
                            --videos_input_path /home/data/input/ \
                            --YOLO_folder_path /home/code/yolov5/runs/detect/ \
                            --white_tray_infos "0.3 0.3 0.7 0.7" \
                            --candidates_file_path /home/data/candidates.txt

# filter_candidates_list
python3 process.py --module filter_candidates_list \
                --input_path /home/data/candidates.txt \
                --output_path /home/data/output/submit.txt