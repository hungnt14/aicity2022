docker run -it --gpus '"device=0"' --shm-size=8g --name <docker container name> \
-v <repo folder>:/home/code/ \
-v <input videos folder>:/home/data/input/ \
-v <output folder that stores submit.txt files>:/home/data/output/ \
<docker image name>

### notes:
# input videos folder should contain videos only (.mp4)

### Example
# docker run -it --gpus '"device=0"' --shm-size=8g --name aicity2022_aiclub_uit \
# -v /home/aiclub/aicity2022/:/home/code/ \
# -v /home/aiclub/data/TestA/:/home/data/input/ \
# -v /home/aiclub/data/output/:/home/data/output/ \
# aicity2022_aiclub_uit