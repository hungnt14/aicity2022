
# NVIDIA AICITY 2022 - Track 4: AICLUB@UIT

  
## Inference
To reproduce our result, first, clone or download this repo.
```
git clone https://github.com/hungnt14/aicity2022
```
When finished, move to `docker` folder to start building docker and run it.
```
cd docker
docker build -t <docker  image  name> .
```

When finishing building the docker images, go ahead to `run_docker.sh` to modify the required information (see that bash file for more details). Then, start the container.

```
bash run_docker.sh
```

After getting into the docker environment, go to `code` folder and install requirements.
```
cd code
pip3 install -r requirements.txt 
```
Now, start to run end-to-end by running the command

```
bash run.sh
```
## Train
See `training` folder for more details

## Contact
If you face with any issues, feel free to contact us at [20520198@gm.uit.edu.vn](mailto:20520198@gm.uit.edu.vn) 
