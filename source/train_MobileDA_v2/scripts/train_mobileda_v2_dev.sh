# resnet50→resnet18
# Ar→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_dev.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'C2I' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/imageclefda/C2I/model_best.pth.tar' \
                                                           -d ImageCLEF \
                                                           -s C \
                                                           -t I \
                                                           --epochs 30 \
                                                           -i 500 \
                                                           --b 64 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0                                  
