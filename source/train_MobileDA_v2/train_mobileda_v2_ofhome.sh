# office-home
# resnet50→resnet18
# Ar→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Ar2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0                                        
# Ar→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Ar2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Ar→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Ar2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Cl2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Cl2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Cl2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Pr2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Pr2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Pr2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Rw2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Rw2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Rw2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
                                                           
# resnet50→resnet34
# Ar→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Ar2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0                                        
# Ar→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Ar2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Ar→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Ar2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Cl2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Cl2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Cl2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Pr2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Pr2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Pr2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Rw2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Rw2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet34' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome/Rw2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# resnet50→resnet34→resnet18
# Ar→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Ar2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0                                        
# Ar→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Ar2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Ar→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Ar2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Ar2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Ar \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Cl2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Cl2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Cl→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Cl2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Cl2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Cl \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Pr2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Pr2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Pr→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Pr2Rw' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Pr2Rw/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Pr \
                                                           -t Rw \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Ar' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Rw2Ar/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Ar \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Cl' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Rw2Cl/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Cl \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# Rw→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34_2_r18/ofhome' \
                                                           --img_root '/content' \
                                                           --note 'Rw2Pr' \
                                                           --t_arch 'resnet50' \
                                                           --s_arch 'resnet18' \
                                                           --t-model-param '/content/drive/MyDrive/results/MobileDA_v2/r50_2_r34/ofhome/Rw2Pr/model_best.pth.tar' \
                                                           -d Officehome \
                                                           -s Rw \
                                                           -t Pr \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0                                                           
