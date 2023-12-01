# DH-Net: Image Registration for Heterologous SAR and Optical Images by using Detection and Description of Local Features

## Requirements

We use PyTorch 1.10 whit Python 3.8, later version should also be compatible. Please refer to requirements.txt for other dependencies.

If you are using conda, you may configure the environment as:

```bash
conda env create -f environment.yml
conda activate match
```

## Getting started

Clone the repo:
```bash
git clone https://github.com/bend1031/DH-net.git && \
```
## Downloading the weights


The off-the-shelf weights can be downloaded by running:

```bash
mkdir weights/d2
wget https://dsmn.ml/files/d2-net/d2_ots.pth -O weights/d2/d2_ots.pth
wget https://dsmn.ml/files/d2-net/d2_tf.pth -O weights/d2/d2_tf.pth
wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O weights/d2/d2_tf_no_phototourism.pth
```

and download model weights from [here](https://drive.google.com/file/d/1Ca0WmKSSt2G6P7m8YAOlSAHEFar_TAWb/view?usp=sharing)

extract weights by

```bash
tar -xvf weights.tar.gz
```

the structure of weights is as follows:

```bash
weights
    ├─d2
    ├─sg
    │  ├─root
    │  └─sp
    ├─sgm
    │  ├─root
    │  └─sp
    └─sp
```
## Matching Pairs

A quick demo for image matching can be called by:

1. change path1 and path2 in main_single.py
2. run main_single.py
```bash
python main_single.py
```

## Evaluation on SOPatch

### download SOPatch dataset

https://pan.baidu.com/s/12yIheGOg6JTTsYAQfX7Pfg?pwd=1234

the structure of dataset is as follows:

```bash
DATASETS
─SOPatch
  ├─OSdataset
  │  ├─test
  │  │  ├─opt
  │  │  └─sar
  │  ├─train
  │  │  ├─opt
  │  │  └─sar
  │  └─val
  │      ├─opt
  │      └─sar
  ├─SEN1-2
  │  ├─test
  │  │  ├─opt
  │  │  └─sar
  │  ├─train
  │  │  ├─opt
  │  │  └─sar
  │  └─val
  │      ├─opt
  │      └─sar
  └─WHU-SEN-City
      ├─test
      │  ├─opt
      │  └─sar
      ├─train
      │  ├─opt
      │  └─sar
      └─val
          ├─opt
          └─sar
```

Run the following command to evaluate the model on SOPatch dataset:

```bash
python main_multi_sopatch.py -m
```

## BibTeX

Thanks for kornia.

Thanks a lot for the great works of the following papers.

Part of the code is borrowed or ported from

SGMNet

D2-Net

LoFTR

SuperPoint

SuperGlue

LightGlue

Disk

Please also cite these works if you find the corresponding code useful.

If you use this code in your project, please cite the following paper:

```bibtex
@InProceedings{Dusmanu2019CVPR,
    author = {Dusmanu, Mihai and Rocco, Ignacio and Pajdla, Tomas and Pollefeys, Marc and Sivic, Josef and Torii, Akihiko and Sattler, Torsten},
    title = {{D2-Net: A Trainable CNN for Joint Detection and Description of Local Features}},
    booktitle = {Proceedings of the 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year = {2019},
}
@article{chen2021sgmnet,
  title={Learning to Match Features with Seeded Graph Matching Network},
  author={Chen, Hongkai and Luo, Zixin and Zhang, Jiahui and Zhou, Lei and Bai, Xuyang and Hu, Zeyu and Tai, Chiew-Lan and Quan, Long},
  journal={International Conference on Computer Vision (ICCV)},
  year={2021}
}
```
