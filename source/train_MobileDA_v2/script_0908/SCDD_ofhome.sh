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
                                            --temperature 2.5 
# 2. SCDD resnet50→resnet34
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 3. SCDD resnet34→resnet18                                                           
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 4. SCDD resnet50→resnet18
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Cl \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Ar→Pr
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50' \
                                            --img_root '/content' \
                                            --note 'Ar2Pr' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Ar \
                                            -t Pr \
                                            --epochs 200 \
                                            --lr 0.01 \
                                            -i 1000 \
                                            -b 64 \
                                            --temperature 2.5 
# 2. SCDD resnet50→resnet34
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Pr \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 3. SCDD resnet34→resnet18                                                           
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Ar2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Pr \
                                                           --epochs 100 \                                                                                                                  
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 4. SCDD resnet50→resnet18
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Ar2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Ar \
                                                           -t Pr \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Cl→Pr
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50' \
                                            --img_root '/content' \
                                            --note 'Cl2Pr' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Cl \
                                            -t Pr \
                                            --epochs 200 \
                                            --lr 0.01 \
                                            -i 1000 \
                                            -b 64 \
                                            --temperature 2.5 
# 2. SCDD resnet50→resnet34
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Cl2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Pr \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 3. SCDD resnet34→resnet18                                                           
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Pr' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Cl2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Pr \
                                                           --epochs 100 \                                                                                                                  
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 4. SCDD resnet50→resnet18
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Cl2Pr/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Pr \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Cl→Ar
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50' \
                                            --img_root '/content' \
                                            --note 'Cl2Ar' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Cl \
                                            -t Ar \
                                            --epochs 200 \
                                            --lr 0.01 \
                                            -i 1000 \
                                            -b 64 \
                                            --temperature 2.5 
# 2. SCDD resnet50→resnet34
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Cl2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Ar \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 3. SCDD resnet34→resnet18                                                           
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Cl2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Ar \
                                                           --epochs 100 \                                                                                                                  
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 4. SCDD resnet50→resnet18
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Cl2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Cl \
                                                           -t Ar \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Pr→Ar
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50' \
                                            --img_root '/content' \
                                            --note 'Pr2Ar' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Pr \
                                            -t Ar \
                                            --epochs 200 \
                                            --lr 0.01 \
                                            -i 1000 \
                                            -b 64 \
                                            --temperature 2.5 
# 2. SCDD resnet50→resnet34
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Pr2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Ar \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 3. SCDD resnet34→resnet18                                                           
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Pr2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Ar \
                                                           --epochs 100 \                                                                                                                  
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 4. SCDD resnet50→resnet18
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Pr2Ar/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Ar \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# Pr→Cl
# 1. pre domain adaptation teacher model
!CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50' \
                                            --img_root '/content' \
                                            --note 'Pr2Cl' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Pr \
                                            -t Cl \
                                            --epochs 200 \
                                            --lr 0.01 \
                                            -i 1000 \
                                            -b 64 \
                                            --temperature 2.5 
# 2. SCDD resnet50→resnet34
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Pr2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Cl \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 3. SCDD resnet34→resnet18                                                           
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet34' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50_2_r34/Pr2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Cl \
                                                           --epochs 100 \                                                                                                                  
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0
# 4. SCDD resnet50→resnet18
!CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50_2_r18' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/SCDD/ofhome/r50/Pr2Cl/model_best.pth.tar' \
                                                           -d OfficeHome \
                                                           -s Pr \
                                                           -t Cl \
                                                           --epochs 100 \
                                                           -i 500 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 8.0                                                                                                                                                                                                                                                                                       