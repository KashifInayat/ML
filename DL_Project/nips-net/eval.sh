# This script is used for evaluating.
for i in "32 outputs/1/ckpt_best.pth" "32 outputs/2/ckpt_best.pth" "32 outputs/3/ckpt_best.pth" "8 outputs/4/ckpt_best.pth" "4 outputs/5/ckpt_best.pth" "3 outputs/6/ckpt_best.pth" "2 outputs/7/ckpt_best.pth"
do
    set -- $i
    echo "Evaluating the $1 - bit model, checkpoint: $2" 
    python main.py --dataset imagenette --arch resnet18 --data_root ~/data/imagenette2/ --bit $1 --init_from $2 -e

done