# This script is used to train 3 FP32 models
for i in "1 0.1 1e-4 64" "2 0.1 1e-4 64"  "3 0.1 1e-4 64"
do
    set -- $i
    python main.py --train_id $1 --lr $2 --wd $3 --batch-size  $4 --dataset imagenette --arch resnet18 --epoch 90 --bit 32 --data_root /soc_local/data/imagenette2 
done
