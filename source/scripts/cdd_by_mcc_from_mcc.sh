# Office31
# A2W
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/SCDD/r50/from_mcc' \
--img_root '/content' \
--note 'A2W' \
--arch 'resnet50' \
-d Office31 \
-s A \
-t W \
-b 64 \

# D2W
# W2D
# A2D
# D2A
# W2A

# OfficeHome
# Ar2Cl
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --save_root '/content/drive/MyDrive/SCDD/of31/SCDD/r50_2_alex/from_mcc/A2W/' \
--img_root '/content' \
--note 'Ar2Cl' \
--arch 'alexnet' \
--model-param '/content/drive/MyDrive/SCDD/of31/SCDD/r50_2_alex/from_mcc/A2W/model_best.pth.tar' \
-d Office31 \
-s Ar \
-t Cl
# Ar2Pr
# Ar2Rw
# Cl2Ar
# Cl2Pr
# Cl2Ar
# Pr2Ar
# Pr2Cl
# Pr2Rw
# Rw2Ar
# Rw2Cl
# Rw2Pr
