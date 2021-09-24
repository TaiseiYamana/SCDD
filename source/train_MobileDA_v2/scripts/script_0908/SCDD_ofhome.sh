# Ar→Cl
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50' \
                                            --img_root '/content' \
                                            --note 'Ar2Cl' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Ar \
                                            -t Cl \
                                            --epochs 200 \
                                            --lr 0.01 \
                                            -i 1000 \
                                            -b 64 \
                                            --temperature 2.5 \
                                            #--model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Cl_lr/checkpoint.pth.tar' \
                                            #--check_point True
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           -i 500 \
                                                           -b 64 \
                                                           --epochs 200 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0 
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cr' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           -i 500 \
                                                           -b 64 \
                                                           --epochs 200 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           -i 500 \
                                                           -b 64 \
                                                           --epochs 200 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0                                                            
                                                           
