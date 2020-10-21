# This script is used to train LSQ quantized models. 
# format (train_id, lr, weight_decay, bitwidth, num_epochs)
for i in "4 0.001 1e-4 8 1" "5 0.01 1e-4 4 90" "6 0.01 0.5e-4 3 90" "7 0.01 0.25e-4 2 90"
do
    set -- $i
    python main.py --train_id $1 --lr $2 --wd $3 --batch-size  64 --dataset imagenette --arch resnet18 --bit $4 --epoch $5  --data_root /soc_local/data/imagenette2 --init_from outputs/1/ckpt_best.pth
done
