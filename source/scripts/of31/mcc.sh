# Office31
# A2W
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/mcc' \
--img_root '/content' \
--note 'A2W' \
--arch 'alexnet' \
-d Office31 \
-s A \
-t W \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

##########################################################################################
# D2W
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/mcc' \
--img_root '/content' \
--note 'D2W' \
--arch 'alexnet' \
-d Office31 \
-s D \
-t W \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

##########################################################################################
# W2D
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/mcc' \
--img_root '/content' \ß
--note 'W2D' \
--arch 'alexnet' \
-d Office31 \
-s W \
-t D \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

##########################################################################################
# A2D
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/mcc' \
--img_root '/content' \
--note 'A2D' \
--arch 'alexnet' \
-d Office31 \
-s A \
-t D \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

##########################################################################################
# D2A
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/mcc' \
--img_root '/content' \
--note 'D2A' \
--arch 'alexnet' \
-d Office31 \
-s D \
-t A \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.

##########################################################################################
# W2A
CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/mcc' \
--img_root '/content' \
--note 'W2A' \
--arch 'alexnet' \
-d Office31 \
-s W \
-t A \
-b 64 \
-i 500 \
--temperature 2. \
--trade_off 1.
