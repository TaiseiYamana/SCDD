# Stepwise Cross Domain Distillation (SCDD)

<div align="center">
    <img src=".github/SCDD.jpg", width="900">
</div>

----------------------------------------------------------------------------------------------------

# Contents
1. [Introduction](#Introduction)
2. [About this repository's folders and files](https://github.com/TaiseiYamana/SCDD/blob/main/README.md#about-this-repositorys-folders-and-files)
3. [Experiment](#Experiment)
4. [Citation](#Citation)
5. [Acknowledgements](#Acknowledgements)

## Introduction

SCDD is UDA method for lightweight model. Lightweight model can be trained by cross domain distillation. This method is a cross-domain distillation focused on improving MobileDA.
This research was published at [APRIS2021](http://sigemb.jp/APRIS/2021/) and paper can be downloaded at [IPSJ](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=216177&item_no=1&page_id=13&block_id=8).

## About this repository's folders and files
Name | Description |
:----- | ----- |
common | Datasets, model definitions, code needed for analysis, etc. from TLlib |
dalib | UDA loss definitions from TLlib |
examples | Sample executable code of TLlib |
kd_losses |　KD loss definitions from KD-Zoo |
param_search | Codes for parameter search using ray|
pycm | Library for computing confusion matrices from pycm|
source |Source code for SCDD|
pseudo_labeling.py | Codes that generate pseudo labels|
split_dataset.py | Codes that splits a dataset into for training and for testing |
utils.py | Loss, accuracy counter from KD-Zoo|


## Experiment
Clone this repository and move the current directory to `/source`.

```
$ git clone https://github.com/TaiseiYamana/SCDD.git
$ cd SCDD/source
```
### Training
Run the shell script file in `/source/scripts/` to start the training process. You can simply specify the hyper-parameters listed in `/source/xxx.py` or manually change them.

The shell script has a data path defined for execution on google colab. You can extract and run `SCDD.ipynb` on google colab for immediate experimentation, but please redefine the data path according to your personal usage.

##### Source only

Normal training with labeled source domain datasets.

Mdoel : `AlexNet`, Dataset : `Office-31`
```
$ bash scripts/of31/source_only.sh
```

##### Cross Domain Distillation (cdd)

The teacher model is pre-domain adapted by MCC, which is then used to train the student model by cross-domain distillation.

Student model : `AlexNet`, Teacher model : `ResNet50`,\
Dataset : `Office-31`
```
$ bash scripts/of31/cdd_by_mcc_from_mcc.sh
```

#### Other methods
##### Deep-CORAL
paper : https://arxiv.org/abs/1607.01719
```
$ bash scripts/of31/dcoral.sh
```
##### MCC (Minimum Class Confusion)
paper : https://arxiv.org/abs/1912.03699
```
$ bash scripts/of31/mcc.sh
```
##### MobileDA
paper : https://ieeexplore.ieee.org/document/9016215
```
$ bash scripts/of31/mobileda.sh
```

#### Stepwise Cross Domain Distillation (SCDD) Experiment
Conduct cross domain distillation via a Teacher Assistant which is mid-size between teacher and student.

Student model : `ResNet18`, Teacher model : `ResNet50`, Teacher Assistant (TA) : `ResNet34`\
Dataset : `Office-home`

##### SCDD

```
$ bash scripts/ofhome/scdd.h
```

##### SCDD w/o TA

```
$ bash scripts/ofhome/scdd_nota.h
```


### Analysis
Run the '/source/analysis.py' to visualize the cross-domain feature representation of the trained model using T-SNE. At the same time, confusion matrix for the target domain is created.

A sample of the code can be run with analysis.sh.

Mdoel : `AlexNet`, Dataset : `Office-31`, Domain Shift : `A→W`\
Comparison method : `source_only`,
```
$ bash scripts/of31/analysis.sh
```

- Display of T-SNE

Blue : source domain (A), Red : target domain (W) \
(Left : source_only, Right : cdd_by_mcc_from_mcc)
<div>
    <tr>
    <td><img src=".github/tsne_source_only.png", width="300"></td>
    <td><img src=".github/tsne_cdd_by_mcc_from_mcc.png", width="300">
    </td>
    </tr>
</div>

<p style="text-indent:1em;"></p>

- Display of Confusion Matrix

(Left : source_only, Right : cdd_by_mcc_from_mcc)
<div>
    <tr>
    <td><img src=".github/cm_source_only.png", width="300"></td>
    <td><img src=".github/cm_cdd_by_mcc_from_mcc.png", width="300">
    </td>
    </tr>
</div>

## Requirements
- python 3.7
- pytorch 1.10.0
- torchvision  0.11.1

## Citation
Please cite these papers in your publications if it helps your research.
```
@inproceedings{weko_216177_1,
   author	 = "Taisei,Yamana and Yuko,Hara-Azumi",
   title	 = "Edge Domain Adaptation through Stepwise Cross-Domain Distillation",
   booktitle	 = "Proceedings of Asia Pacific Conference on Robot IoT System Development and Platform",
   year 	 = "2022",
   volume	 = "2021",
   number	 = "",
   pages	 = "1--7",
   month	 = "jan"
}
```
Links to the paper:
- [Edge Domain Adaptation through Stepwise Cross-Domain Distillation](https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main&active_action=repository_view_main_item_detail&item_id=216177&item_no=1&page_id=13&block_id=8)

## Acknowledgements
[TLlib](https://github.com/thuml/Transfer-Learning-Library) \
[Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo) \
[pycm](https://github.com/sepandhaghighi/pycm) \
[ray](https://github.com/ray-project/ray) \
[In Defense of Pseudo-Labeling](https://github.com/nayeemrizve/ups)
