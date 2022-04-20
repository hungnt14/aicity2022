### Training
Our training datasets (and weights also) are put at this [Drive folder](https://drive.google.com/drive/folders/1ZyRNhNctXk_J9K2YtWP9cD_dmfk3WcOQ?usp=sharing)

Download `Auto-retail-syndata-release.zip` and `data_train_yolov5.zip` from the above Drive folder.

Then create `data` and `checkpoints` folder 

Next, extract `Auto-retail-syndata-release.zip` and `data_train_yolov5` to `data` folder.

After finishing extracting data, moving to the `docker` folder.

+ Build docker image by using the following command

```
docker build -t <docker  image  name> .
```
+ Then, run the bash file to run the docker image.

```
bash run_docker_train.sh
```

+ After getting into the docker, start installing required libraries and training
```
cd /home/code/
pip3 install -r requirements.txt
bash run_train.sh
```