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
--lam 1. \
--mu 1. \
--mcc_temp 2. \
--st_temp 4.

# D2W
# pretrain teacher
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50/mcc' \
--img_root '/content' \
--note 'D2W' \
--arch 'resnet50' \
-d Office31 \
-s D \
-t W \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc' \
--img_root '/content' \
--note 'D2W' \
--t_arch 'resnet50' \
--s_arch 'alexnet' \
--t_model_param '/content/drive/MyDrive/SCDD/of31/r50/mcc/D2W/model_best.pth.tar' \
-d Office31 \
-s D \
-t W \
-b 64 \
-i 500 \
--lam 1. \
--mu 1. \
--mcc_temp 2. \
--st_temp 4.

# W2D
# pretrain teacher
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50/mcc' \
--img_root '/content' \
--note 'W2D' \
--arch 'resnet50' \
-d Office31 \
-s W \
-t D \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc' \
--img_root '/content' \
--note 'W2D' \
--t_arch 'resnet50' \
--s_arch 'alexnet' \
--t_model_param '/content/drive/MyDrive/SCDD/of31/r50/mcc/W2D/model_best.pth.tar' \
-d Office31 \
-s W \
-t D \
-b 64 \
-i 500 \
--lam 1. \
--mu 1. \
--mcc_temp 2. \
--st_temp 4.

# A2D
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50/mcc' \
--img_root '/content' \
--note 'A2D' \
--arch 'resnet50' \
-d Office31 \
-s A \
-t D \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc' \
--img_root '/content' \
--note 'A2D' \
--t_arch 'resnet50' \
--s_arch 'alexnet' \
--t_model_param '/content/drive/MyDrive/SCDD/of31/r50/mcc/A2D/model_best.pth.tar' \
-d Office31 \
-s A \
-t D \
-b 64 \
-i 500 \
--lam 1. \
--mu 1. \
--mcc_temp 2. \
--st_temp 4.

# D2A
# pretrain teacher
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50/mcc' \
--img_root '/content' \
--note 'D2A' \
--arch 'resnet50' \
-d Office31 \
-s D \
-t A \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc' \
--img_root '/content' \
--note 'D2A' \
--t_arch 'resnet50' \
--s_arch 'alexnet' \
--t_model_param '/content/drive/MyDrive/SCDD/of31/r50/mcc/D2A/model_best.pth.tar' \
-d Office31 \
-s D \
-t A \
-b 64 \
-i 500 \
--lam 1. \
--mu 1. \
--mcc_temp 2. \
--st_temp 4.

# W2A
# pretrain teacher
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50/mcc' \
--img_root '/content' \
--note 'W2A' \
--arch 'resnet50' \
-d Office31 \
-s W \
-t A \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

CUDA_VISIBLE_DEVICES=0 python3 cdd_with_mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/r50_2_alex/from_mcc' \
--img_root '/content' \
--note 'W2A' \
--t_arch 'resnet50' \
--s_arch 'alexnet' \
--t_model_param '/content/drive/MyDrive/SCDD/of31/r50/mcc/W2A/model_best.pth.tar' \
-d Office31 \
-s W \
-t A \
-b 64 \
-i 500 \
--lam 1. \
--mu 1. \
--mcc_temp 2. \
--st_temp 4.
