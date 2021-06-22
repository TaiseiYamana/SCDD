# Office31
# resnet50
# A→W
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_A2W_r50' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 1024 \
                                                         -d Office31 \
                                                         -s A \
                                                         -t W \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# A→D                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_A2D_r50' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 1024 \
                                                         -d Office31 \
                                                         -s A \
                                                         -t D \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# W→A                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_W2A_r50' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 1024 \
                                                         -d Office31 \
                                                         -s W \
                                                         -t A \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
#W→D                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_W2D_r50' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 1024 \
                                                         -d Office31 \
                                                         -s W \
                                                         -t D \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# D→A                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_D2A_r50' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 1024 \
                                                         -d Office31 \
                                                         -s D \
                                                         -t A \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# D→W                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_D2W_r50' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 1024 \
                                                         -d Office31 \
                                                         -s D \
                                                         -t W \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# resnet34
# A→W
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_A2W_r34' \
                                                         -a 'resnet34' \
                                                         --bottleneck-dim 512 \
                                                         -d Office31 \
                                                         -s A \
                                                         -t W \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# A→D                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_A2D_r34' \
                                                         -a 'resnet34' \
                                                         --bottleneck-dim 512 \
                                                         -d Office31 \
                                                         -s A \
                                                         -t D \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# W→A                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_W2A_r34' \
                                                         -a 'resnet34' \
                                                         --bottleneck-dim 512 \
                                                         -d Office31 \
                                                         -s W \
                                                         -t A \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
#W→D                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_W2D_r34' \
                                                         -a 'resnet34' \
                                                         --bottleneck-dim 512 \
                                                         -d Office31 \
                                                         -s W \
                                                         -t D \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# D→A                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_D2A_r34' \
                                                         -a 'resnet34' \
                                                         --bottleneck-dim 512 \
                                                         -d Office31 \
                                                         -s D \
                                                         -t A \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
# D→W                                                         
CUDA_VISIBLE_DEVICES=0 python3 pretrain_teacher_model.py --save_root '/content/drive/MyDrive/results/PT' \
                                                         --img_root '/content' \
                                                         --note 'pt_of31_D2W_r34' \
                                                         -a 'resnet50' \
                                                         --bottleneck-dim 512 \
                                                         -d Office31 \
                                                         -s D \
                                                         -t W \
                                                         --epochs 20 \
                                                         -i 500 \
                                                         --seed 2 \
                                                         --temperature 2.5
