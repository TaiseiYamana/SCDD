# Office31
# A2W
CUDA_VISIBLE_DEVICES=0 python3 analysis.py --save_root '/content/drive/MyDrive/SCDD/of31/analysis' \
--img_root '/content' \
--note 'A2W/source_only' \
--arch 'alexnet' \
--model_param '/content/drive/MyDrive/SCDD/of31/alex/source_only/A2W/model_best.pth.tar' \
-d Office31 \
-s A \
-t W

CUDA_VISIBLE_DEVICES=0 python3 analysis.py --save_root '/content/drive/MyDrive/SCDD/of31/analysis' \
--img_root '/content' \
--note 'A2W/cdd_with_mcc_from_mcc' \
--arch 'alexnet' \
--model_param '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc/A2W/model_best.pth.tar' \
-d Office31 \
-s A \
-t W

# D2W
# W2D
# A2D
# D2A
# W2A

# OfficeHome
# Ar2Cl
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