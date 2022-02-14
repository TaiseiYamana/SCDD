# imgclefda
# C→I
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/imgclef/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'C2I' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/imgclef/r50/C2I/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s C \
                                                           -t I \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'C2I' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/C2I/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s C \
                                                           -t I \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'C2I' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/C2I/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s C \
                                                           -t I \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0                                                                                                                                                                     