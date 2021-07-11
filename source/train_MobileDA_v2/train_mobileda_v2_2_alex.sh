# office-31
# resnet34→arexnet
# A→W
CUDA_VISIBLE_DEVICES=0 python3 train_mobileda_v2_2_alex.py --save_root '/content/drive/MyDrive/results/MobileDA_v2/r50_2_arex/of31' \
                                                           --img_root '/content' \
                                                           --note 'A2W' \
                                                           --t_arch 'resnet50' \
                                                           --t-model-param '/content/drive/MyDrive/results/PT/MCC/r34/of31_A2W/model_best.pth.tar' \
                                                           -d Office31 \
                                                           -s A \
                                                           -t W \
                                                           --epochs 20 \
                                                           -i 500 \
                                                           --seed 2 \
                                                           --mcc_temp 2.5 \
                                                           --st_temp 4.0
# D→W
 
# W→D
     
# A→D
     
# D→A

# W→A
                                        
