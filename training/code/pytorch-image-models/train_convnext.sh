bash /home/code/pytorch-image-models/distributed_train.sh 1 \
    /home/data/ \
    --dataset ImageFolder --model convnext_large --epochs 10 --batch-size 4 --num-classes 116 \
    --output /home/checkpoints/convnext/