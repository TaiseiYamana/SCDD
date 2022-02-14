# Office31
# resnet50
# Ar→Cl
CUDA_VISIBLE_DEVICES=0 python3 raytune_mcc.sh --img_root '/content' \
                                            -a 'resnet50' \
                                            -d OfficeHome \
                                            -s Ar \
                                            -t Cl \
                                            -i 500 \
                                            --epochs 10 \
                                            --seed 0 \
                                            --temperature 2.5  
