#office31
# Teacher : resnet50
# A→W
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r34_2_r18' \
                                      --img_root '/content' \
                                      --note 'of31_A2W' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/ResNet18/r50_2_r34/of31_A2W/model_best.pth.tar' \
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
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r34_2_r18' \
                                      --img_root '/content' \
                                      --note 'of31_A2D' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/ResNet18/r50_2_r34/of31_A2D/model_best.pth.tar' \
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
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r34_2_r18' \
                                      --img_root '/content' \
                                      --note 'of31_W2A' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/ResNet18/r50_2_r34/of31_W2A/model_best.pth.tar' \
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
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r34_2_r18' \
                                      --img_root '/content' \
                                      --note 'of31_W2D' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/ResNet18/r50_2_r34/of31_W2D/model_best.pth.tar' \
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
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r34_2_r18' \
                                      --img_root '/content' \
                                      --note 'of31_D2A' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/ResNet18/r50_2_r34/of31_D2A/model_best.pth.tar' \
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
CUDA_VISIBLE_DEVICES=0 python3 distillation_to_r18.py --save_root '/content/drive/MyDrive/results/ResNet18/r50_2_r34_2_r18' \
                                      --img_root '/content' \
                                      --note 'of31_D2W' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/ResNet18/r50_2_r34/of31_D2W/model_best.pth.tar' \
                                      --s_arch 'resnet18' \
                                      -d Office31 \
                                      -s D \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
