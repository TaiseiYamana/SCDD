# Office31
# A2W
#CUDA_VISIBLE_DEVICES=0 python3 source_only.py --save_root '/content/drive/MyDrive/SCDD/of31/alex/source_only' \
#--img_root '/content' \
#--note 'A2W' \
#--arch 'resnet50' \
#-d Office31 \
#-s A \
#-t W \
#-b 64 \
#-i 500

CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/source_only' \
--img_root '/content' \
--note 'A2W' \
--arch 'alexnet' \
-d Office31 \
-s A \
-t W \
-b 64 \
-i 500 \
--trade_off 0.

##########################################################################################
# D2W
#CUDA_VISIBLE_DEVICES=0 python3 source_only.py --save_root '/content/drive/MyDrive/SCDD/of31/alex/source_only' \
#--img_root '/content' \
#--note 'D2W' \
#--arch 'resnet50' \
#-d Office31 \
#-s D \
#-t W \
#-b 64 \
#-i 500

CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/source_only' \
--img_root '/content' \
--note 'D2W' \
--arch 'alexnet' \
-d Office31 \
-s D \
-t W \
-b 64 \
-i 500 \
--trade_off 0.

##########################################################################################
# W2D
#CUDA_VISIBLE_DEVICES=0 python3 source_only.py --save_root '/content/drive/MyDrive/SCDD/of31/alex/source_only' \
#--img_root '/content' \
#--note 'W2D' \
#--arch 'resnet50' \
#-d Office31 \
#-s W \
#-t D \
#-b 64 \
#-i 500

CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/source_only' \
--img_root '/content' \
--note 'W2D' \
--arch 'alexnet' \
-d Office31 \
-s W \
-t D \
-b 64 \
-i 500 \
--trade_off 0.

##########################################################################################
# A2D
#CUDA_VISIBLE_DEVICES=0 python3 source_only.py --save_root '/content/drive/MyDrive/SCDD/of31/alex/source_only' \
#--img_root '/content' \
#--note 'A2D' \
#--arch 'resnet50' \
#-d Office31 \
#-s A \
#-t D \
#-b 64 \
#-i 500

CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/source_only' \
--img_root '/content' \
--note 'A2D' \
--arch 'alexnet' \
-d Office31 \
-s A \
-t D \
-b 64 \
-i 500 \
--trade_off 0.

##########################################################################################
# D2A
#CUDA_VISIBLE_DEVICES=0 python3 source_only.py --save_root '/content/drive/MyDrive/SCDD/of31/alex/source_only' \
#--img_root '/content' \
#--note 'D2A' \
#--arch 'resnet50' \
#-d Office31 \
#-s D \
#-t A \
#-b 64 \
#-i 500

CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/source_only' \
--img_root '/content' \
--note 'D2A' \
--arch 'alexnet' \
-d Office31 \
-s D \
-t A \
-b 64 \
-i 500 \
--trade_off 0.

##########################################################################################
# W2A
#CUDA_VISIBLE_DEVICES=0 python3 source_only.py --save_root '/content/drive/MyDrive/SCDD/of31/alex/source_only' \
#--img_root '/content' \
#--note 'W2A' \
#--arch 'resnet50' \
#-d Office31 \
#-s  \W
#-t A \
#-b 64 \
#-i 500

CUDA_VISIBLE_DEVICES=0 python3 mcc.py --save_root '/content/drive/MyDrive/SCDD/of31/alexnet/source_only' \
--img_root '/content' \
--note 'W2A' \
--arch 'alexnet' \
-d Office31 \
-s W \
-t A \
-b 64 \
-i 500 \
--trade_off 0.
