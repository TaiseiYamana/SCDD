# Office31
# resnet50
# Ar→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Ar2Cl' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Ar \
                                            -t Cl \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5                                           
# Ar→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Ar2Pr' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Ar \
                                            -t Pr \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Ar→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Ar2Rw' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Ar \
                                            -t Rw \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Cl→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Cl2Ar' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Cl \
                                            -t Ar \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Cl→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Cl2Pr' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Cl \
                                            -t Pr \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Cl→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Cl2Rw' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Cl \
                                            -t Rw \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Pr→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Pr2Ar' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Pr \
                                            -t Ar \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Pr→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Pr2Cl' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Pr \
                                            -t Cl \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Pr→Rw
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Pr2Rw' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Pr \
                                            -t Rw \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Rw→Ar
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Rw2Ar' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Rw \
                                            -t Ar \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Rw→Cl
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Rw2Cl' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Rw \
                                            -t Cl \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
# Rw→Pr
CUDA_VISIBLE_DEVICES=0 python3 train_mcc.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50/ofhome' \
                                            --img_root '/content' \
                                            --note 'Rw2Pr' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Rw \
                                            -t Pr \
                                            --epochs 30 \
                                            -i 500 \
                                            --seed 0 \
                                            --temperature 2.5
