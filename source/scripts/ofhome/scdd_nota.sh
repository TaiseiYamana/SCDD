# OfficeHome
# Ar2Cl
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Ar2Cl' \
--arch 'resnet50' \
-d OfficeHome \
-s Ar \
-t Cl \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Ar2Cl' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Ar2Cl/model_best.pth.tar' \
-d OfficeHome \
-s Ar \
-t Cl \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Ar2Pr
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Ar2Pr' \
--arch 'resnet50' \
-d OfficeHome \
-s Ar \
-t Pr \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Ar2Pr' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Ar2Pr/model_best.pth.tar' \
-d OfficeHome \
-s Ar \
-t Pr \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Ar2Rw
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Ar2Rw' \
--arch 'resnet18' \
-d OfficeHome \
-s Ar \
-t Rw \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Ar2Rw' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Ar2Rw/model_best.pth.tar' \
-d OfficeHome \
-s Ar \
-t Rw \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Cl2Ar
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Cl2Ar' \
--arch 'resnet50' \
-d OfficeHome \
-s Cl \
-t Ar \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Cl2Ar' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Cl2Ar/model_best.pth.tar' \
-d OfficeHome \
-s Cl \
-t Ar \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Cl2Pr
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Cl2Pr' \
--arch 'resnet18' \
-d OfficeHome \
-s Cl \
-t Pr \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Cl2Pr' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Cl2Pr/model_best.pth.tar' \
-d OfficeHome \
-s Cl \
-t Pr \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Cl2Rw
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Cl2Rw' \
--arch 'resnet50' \
-d OfficeHome \
-s Cl \
-t Rw \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Cl2Rw' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Cl2Rw/model_best.pth.tar' \
-d OfficeHome \
-s Cl \
-t Rw \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Pr2Ar
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Pr2Ar' \
--arch 'resnet50' \
-d OfficeHome \
-s Pr \
-t Ar \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Pr2Ar' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Pr2Ar/model_best.pth.tar' \
-d OfficeHome \
-s Pr \
-t Ar \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Pr2Cl
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Pr2Cl' \
--arch 'resnet50' \
-d OfficeHome \
-s Pr \
-t Cl \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Pr2Cl' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Pr2Cl/model_best.pth.tar' \
-d OfficeHome \
-s Pr \
-t Cl \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Pr2Rw
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Pr2Rw' \
--arch 'resnet50' \
-d OfficeHome \
-s Pr \
-t Rw \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Pr2Rw' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Pr2Rw/model_best.pth.tar' \
-d OfficeHome \
-s Pr \
-t Rw \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Rw2Ar
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Rw2Ar' \
--arch 'resnet50' \
-d OfficeHome \
-s Rw \
-t Ar \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Rw2Ar' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Rw2Ar/model_best.pth.tar' \
-d OfficeHome \
-s Rw \
-t Ar \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Rw2Cl
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Rw2Cl' \
--arch 'resnet50' \
-d OfficeHome \
-s Rw \
-t Cl \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Rw2Cl' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Rw2Cl/model_best.pth.tar' \
-d OfficeHome \
-s Rw \
-t Cl \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.

##########################################################################################
# Rw2Pr
# pre-domain adaptation teacher model (resnet50)
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/r50/mcc' \
--img_root '/content' \
--note 'Rw2Pr' \
--arch 'resnet50' \
-d OfficeHome \
-s Rw \
-t Pr \
-b 64 \
-i 1000 \
--temperature 2.5 \
--trade_off 1.

# cross domain distillation (resnet50->resnet18)
CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/ofhome/SCDD/r50_2_r18' \
--img_root '/content' \
--note 'Rw2Pr' \
--t_arch 'resnet50' \
--s_arch 'resnet18' \
--t_model_param '/content/drive/MyDrive/SCDD/ofhome/r50/mcc/Rw2Pr/model_best.pth.tar' \
-d OfficeHome \
-s Rw \
-t Pr \
-b 64 \
-i 1000 \
--lam 1. \
--mu 1. \
--mcc_temp 2.5 \
--st_temp 8.
