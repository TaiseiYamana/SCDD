# Office31
# resnet18
# A→W
!CUDA_VISIBLE_DEVICES=0 python3 train_base.py --save_root '/content/drive/MyDrive/results/Base' \
                                      --img_root '/content' \
                                      --note 'base_of31_A2W_r18' \
                                      -a 'resnet18' \
                                      -d Office31 \
                                      -s A \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2
# A→D
!CUDA_VISIBLE_DEVICES=0 python3 train_base.py --save_root '/content/drive/MyDrive/results/Base' \
                                      --img_root '/content' \
                                      --note 'base_of31_A2D_r18' \
                                      -a 'resnet18' \
                                      -d Office31 \
                                      -s A \
                                      -t D \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2
# W→A
!CUDA_VISIBLE_DEVICES=0 python3 train_base.py --save_root '/content/drive/MyDrive/results/Base' \
                                      --img_root '/content' \
                                      --note 'base_of31_W2A_r18' \
                                      -a 'resnet18' \
                                      -d Office31 \
                                      -s W \
                                      -t A \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2
# W→D
!CUDA_VISIBLE_DEVICES=0 python3 train_base.py --save_root '/content/drive/MyDrive/results/Base' \
                                      --img_root '/content' \
                                      --note 'base_of31_W2D_r18' \
                                      -a 'resnet18' \
                                      -d Office31 \
                                      -s W \
                                      -t D \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2              
# D→A
!CUDA_VISIBLE_DEVICES=0 python3 train_base.py --save_root '/content/drive/MyDrive/results/Base' \
                                      --img_root '/content' \
                                      --note 'base_of31_D2A_r18' \
                                      -a 'resnet18' \
                                      -d Office31 \
                                      -s D \
                                      -t A \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2                      
# D→W
!CUDA_VISIBLE_DEVICES=0 python3 train_base.py --save_root '/content/drive/MyDrive/results/Base' \
                                      --img_root '/content' \
                                      --note 'base_of31_D2W_r18' \
                                      -a 'resnet18' \
                                      -d Office31 \
                                      -s D \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2                                                   
