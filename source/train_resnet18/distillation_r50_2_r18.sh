#office31
# Teacher : resnet50
# A→W
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r18' \
                                      --img_root '/content' \
                                      --note 'mda_of31_A2W' \
                                      --t_arch 'resnet50' \
                                      --t-bottleneck-dim 1024 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/pt_of31_A2W_r50/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s A \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
# A→D                                      
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r18' \
                                      --img_root '/content' \
                                      --note 'mda_of31_A2D' \
                                      --t_arch 'resnet50' \
                                      --t-bottleneck-dim 1024 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/pt_of31_A2D_r50/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s A \
                                      -t D \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
# W→A                                      
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r18' \
                                      --img_root '/content' \
                                      --note 'mda_of31_W2A' \
                                      --t_arch 'resnet50' \
                                      --t-bottleneck-dim 1024 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/pt_of31_W2A_r50/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s W \
                                      -t A \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
 # W→D                                     
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r18' \
                                      --img_root '/content' \
                                      --note 'mda_of31_W2D' \
                                      --t_arch 'resnet50' \
                                      --t-bottleneck-dim 1024 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/pt_of31_W2D_r50/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s W \
                                      -t D \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
# D→A                                      
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r18' \
                                      --img_root '/content' \
                                      --note 'mda_of31_D2A' \
                                      --t_arch 'resnet50' \
                                      --t-bottleneck-dim 1024 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/pt_of31_D2A_r50/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s D \
                                      -t A \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
# D→W                                      
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r18' \
                                      --img_root '/content' \
                                      --note 'mda_of31_D2W' \
                                      --t_arch 'resnet50' \
                                      --t-bottleneck-dim 1024 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/pt_of31_D2W_r50/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s D \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
