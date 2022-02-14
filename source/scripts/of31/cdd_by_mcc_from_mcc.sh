# Office31
# A2W
# pretrain teacher
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50/mcc' \
--img_root '/content' \
--note 'A2W' \
--arch 'resnet50' \
-d Office31 \
-s A \
-t W \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc' \
--img_root '/content' \
--note 'A2W' \
--t_arch 'resnet50' \
--s_arch 'alexnet' \
--t_model_param '/content/drive/MyDrive/SCDD/of31/r50/mcc/A2W/model_best.pth.tar' \
-d Office31 \
-s A \
-t W \
-b 64 \
-i 500 \
--lam 1. \ # mcc
--mu 1. \ # kd
--mcc_temp 2. \
--st_temp 4.

# D2W
# W2D
# A2D
# D2A
# W2A
