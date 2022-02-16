# office-31
# resnet50→arexnet
# A→W
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'A2W' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/of31/A2W/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s A \
                                                           -t W \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# D→W
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'D2W' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/of31/D2W/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s D \
                                                           -t W \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# W→D
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'W2D' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/of31/W2D/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s W \
                                                           -t D \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0     
# A→D
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'A2D' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/of31/A2D/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s A \
                                                           -t D \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0     
# D→A
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'D2A' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/of31/D2A/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s D \
                                                           -t A \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0 
# W→A
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'W2A' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/of31/W2A/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s W \
                                                           -t A \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0                                         
