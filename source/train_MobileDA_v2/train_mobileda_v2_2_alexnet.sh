# office-31
# resnet34→arexnet
# A→W
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alexnet.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r34_2_arex' \
                                      --img_root '/content' \
                                      --note 'of31_A2W' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_A2W' \
                                      -d Office31 \
                                      -s A \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
# D→W
CUDA_VISIBLE_DEVICES=0 python3 Distillation_from_teacher_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r34_2_arex' \
                                      --img_root '/content' \
                                      --note 'of31_D2W' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_D2W' \
                                      -d Office31 \
                                      -s D \
                                      -t W \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0 
# W→D
CUDA_VISIBLE_DEVICES=0 python3 Distillation_from_teacher_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r34_2_arex' \
                                      --img_root '/content' \
                                      --note 'of31_W2D' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_W2D' \
                                      -d Office31 \
                                      -s w \
                                      -t D \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0      
# A→D
CUDA_VISIBLE_DEVICES=0 python3 Distillation_from_teacher_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r34_2_arex' \
                                      --img_root '/content' \
                                      --note 'of31_A2D' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_A2D' \
                                      -d Office31 \
                                      -s A \
                                      -t D \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0     
# D→A
CUDA_VISIBLE_DEVICES=0 python3 Distillation_from_teacher_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r34_2_arex' \
                                      --img_root '/content' \
                                      --note 'of31_D2A' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_D2A' \
                                      -d Office31 \
                                      -s D \
                                      -t A \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0
# W→A
CUDA_VISIBLE_DEVICES=0 python3 Distillation_from_teacher_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r34_2_arex' \
                                      --img_root '/content' \
                                      --note 'of31_W2A' \
                                      --t_arch 'resnet34' \
                                      --t-bottleneck-dim 512 \
                                      --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_W2A' \
                                      -d Office31 \
                                      -s W \
                                      -t A \
                                      --epochs 20 \
                                      -i 500 \
                                      --seed 2 \
                                      --mcc_temp 2.5 \
                                      --st_temp 4.0                                        
