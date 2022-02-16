# office-home
# Ar→Cl
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Ar→Pr
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Ar2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Pr \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/Ar2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Pr \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Ar2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Pr \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Cl→Ar
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Cl2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Ar \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/Cl2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Ar \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Cl2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Ar \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0 
# Cl→Pr
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Cl2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Pr \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Pr' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/Cl2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Pr \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Cl2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Pr \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Pr→Ar
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Pr2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Ar \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/Pr2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Ar \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Pr2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Ar \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Pr→Cl
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Pr2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Cl \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r34/Pr2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Cl \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0                                                          
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MCC/ofhome/r50/Pr2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Cl \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0                                                                                                                                                                           