FROM nvidia/cuda:11.0-base
ENV DEBIAN_FRONTEND=nonintercative

RUN apt-get update && apt-get -y install libgtk2.0-dev git python3-pip sudo libjpeg-dev zlib1g-dev libgl1 && pip3 install --upgrade pip
RUN pip3 install gdown Pillow torch torchvision matplotlib numpy opencv-python PyYAML requests scipy tqdm natsort timm==0.5.4
RUN cd home && mkdir code && mkdir data && mkdir checkpoints && cd checkpoints && mkdir convnext yolov5

WORKDIR /home