# Image Super Resolution

Collection of experiments in Single Image Super Resolution

## Datasets

[Strange Phenomena and Yōkai](https://www.nichibun.ac.jp/en/db/category/yokaigazou/)

[Japanese Woodblock Print Search](https://ukiyo-e.org/)

## Implemented Models

'default' [Enhanced Deep Residual Networks](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)

'--srrn' [Super Resolution Residual Network](https://arxiv.org/pdf/1609.04802)

## Usage

To run the training mechanism with default settings run `python main.py`

To run the model in test mode, use the `--test` flag 
and specify the directory with pretrained weights and input (low resolution) images

### General Flags

`-n, --n` - number of training epochs

`--seed` - manual random seed

`--batchsize` - batch size

'--lr' - learning rate for training

### Dataset Flags

`--augment` - run auto-augmentation on input dataset and store in input directory

### Test Flags

`--test` - set testing mode to true and generate images with selected model and path to folder with pretrained model weights and input images
