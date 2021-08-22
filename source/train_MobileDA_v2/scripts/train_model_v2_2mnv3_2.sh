# C->I
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobile_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'C2I' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda_2/C2I/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s C \
                                                           -t I \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# C->P
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'C2P' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda_2/C2P/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s C \
                                                           -t P \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# I->C
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'I2C' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda_2/I2C/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s I \
                                                           -t C \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# I->P
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'I2P' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda_2/I2P/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s I \
                                                           -t P \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# P->C
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'P2C' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda_2/P2C/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s P \
                                                           -t C \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# P->I
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'P2I' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda_2/P2I/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s P \
                                                           -t I \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             