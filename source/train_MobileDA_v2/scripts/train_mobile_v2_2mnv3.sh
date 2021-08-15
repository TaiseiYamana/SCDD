# C->I
# r18->mnv3s
!CUDA_VISIBLE_DEVICES=0 python3 train_mobile_v2_2_mnv3.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18_2_mnv3s/imageclefda_2' \
                                                           --img_root '/content' \
                                                           --note 'C2I' \
                                                           --t_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_r_18/imageclefda_2/C2I/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s C \
                                                           -t I \
                                                           --epochs 30 \
                                                           --lr 0.01 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0