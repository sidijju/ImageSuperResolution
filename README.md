# Image Super Resolution

Collection of experiments in Single Image Super Resolution

## Datasets

[Strange Phenomena and Yōkai](https://www.nichibun.ac.jp/en/db/category/yokaigazou/)

[Japanese Woodblock Print Search](https://ukiyo-e.org/)

## Implemented Models

[Enhanced Deep Residual Networks](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)

[Super Resolution Residual Network](https://arxiv.org/pdf/1609.04802)

## Usage

To run the training mechanism with default settings run `python train.py`

For testing, run `python test.py` and pass in the path to pretrained model weights and the input image directory

### Train Flags

`-n, --n` - number of training steps (gradient updates)

`--seed` - manual random seed

`--batchsize` - batch size

`--lr` - learning rate for training

`--augment` - run auto-augmentation on input dataset and store in input directory

`--srrn` - train Super Resolution Residual Network (default is EDSR)

### Test Flags

`--model` - path to pretrained model weights

`--images` - path to input image folder