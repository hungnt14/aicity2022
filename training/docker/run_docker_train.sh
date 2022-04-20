docker run -it --gpus '"device=0"' --shm-size=8g --name <your docker image name> \
-v <full path to the current training folder>:/home/ \
<your docker image name>

## Note:
# Docker require a full path to the current directory for mounting
### Example
# docker run -it --gpus '"device=0"' --shm-size=8g --name aicity2022_aiclub_uit_train \
# -v /home/aiclub/aicity2022/training/:/home/ \
# aicity2022_aiclub_uit_train