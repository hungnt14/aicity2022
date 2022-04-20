FROM nvidia/cuda:11.0-base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install libgtk2.0-dev git python3-pip sudo libjpeg-dev zlib1g-dev libgl1 && pip3 install --upgrade pip && pip3 install gdown

WORKDIR /home
RUN mkdir code && mkdir weights && mkdir data && cd data && mkdir input && mkdir output

RUN cd weights && gdown 18vbOfHORarya6i7544fcdT24kiOj9hww && gdown 1du4tFAwMdXelwAxDWCQ5zMNsQlI3AFuM