# Basic Fine-Grained Sketch-Based Image Retrieval

This repository is built for traditional Fine-Grained Sketch-Based Image Retrieval

## Dataset

QMUL-Shoe-V2 and QMUL-Chair-V2 dataset will be used. Please follow this [link](https://www.kaggle.com/datasets/tuanhuu17052002/fg-sbir-dataset/data) to download dataset.

## Training

This tutorial will show how to train with Dataset QMUL-Shoe-V2. Train with QMUL-Chair-V2 is the same.

Firstly, clone the repository from github

```console
!git clone https://github.com/huutuan1705/Base-FG-SBIR.git
```

Install libraries from requirements.txt

```console
cd Base-FG-SBIR
!pip install -r requirements.txt
```

Training with QMUL-Shoe-V2 Dataset. You can choose another dataset to training.

```console
cd baseline_fg_sbir
!python main.py --root_dir /kaggle/input/fg-sbir-dataset --batch_size 48
```

