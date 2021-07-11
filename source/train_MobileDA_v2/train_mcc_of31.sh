# Office31
# resnet50
# A→W
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/of31' \
                                                         --img_root '/content' \
                                                         --note 'A2W' \
                                                         -a 'resnet50' \
                                                         -d Office31 \
                                                         -s A \
                                                         -t W \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
                                                        
# D→W
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/of31' \
                                                         --img_root '/content' \
                                                         --note 'D2W' \
                                                         -a 'resnet50' \
                                                         -d Office31 \
                                                         -s D \
                                                         -t W \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# W→D
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/of31' \
                                                         --img_root '/content' \
                                                         --note 'W2D' \
                                                         -a 'resnet50' \
                                                         -d Office31 \
                                                         -s W \
                                                         -t D \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# A→D
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/of31' \
                                                         --img_root '/content' \
                                                         --note 'A2D' \
                                                         -a 'resnet50' \
                                                         -d Office31 \
                                                         -s A \
                                                         -t D \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# D→A
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/of31' \
                                                         --img_root '/content' \
                                                         --note 'D2A' \
                                                         -a 'resnet50' \
                                                         -d Office31 \
                                                         -s D \
                                                         -t A \
                                                         --epochs  \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5 
# W→A
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/of31' \
                                                         --img_root '/content' \
                                                         --note 'W2A' \
                                                         -a 'resnet50' \
                                                         -d Office31 \
                                                         -s W \
                                                         -t A \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5                                                         
