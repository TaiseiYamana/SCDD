# Office31
# A2W
CUDA_VISIBLE_DEVICES=0 python3 dcoral.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/dcoral' \
--img_root '/content' \
--note 'A2W' \
--arch 'alexnet' \
-d Office31 \
-s A \
-t W \
-i 500 \
--trade_off 0.75

# D2W
CUDA_VISIBLE_DEVICES=0 python3 dcoral.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/dcoral' \
--img_root '/content' \
--note 'D2W' \
--arch 'alexnet' \
-d Office31 \
-s D \
-t W \
-i 500 \
--trade_off 0.75

# W2D
CUDA_VISIBLE_DEVICES=0 python3 dcoral.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/dcoral' \
--img_root '/content' \
--note 'W2D' \
--arch 'alexnet' \
-d Office31 \
-s W \
-t D \
-i 500 \
--trade_off 0.75

# A2D
CUDA_VISIBLE_DEVICES=0 python3 dcoral.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/dcoral' \
--img_root '/content' \
--note 'A2D' \
--arch 'alexnet' \
-d Office31 \
-s A \
-t D \
-i 500 \
--trade_off 0.75

# D2A
CUDA_VISIBLE_DEVICES=0 python3 dcoral.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/dcoral' \
--img_root '/content' \
--note 'D2A' \
--arch 'alexnet' \
-d Office31 \
-s D \
-t A \
-i 500 \
--trade_off 0.75

# W2A
CUDA_VISIBLE_DEVICES=0 python3 dcoral.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/dcoral' \
--img_root '/content' \
--note 'W2A' \
--arch 'alexnet' \
-d Office31 \
-s W \
-t A \
-i 500 \
--trade_off 0.75
