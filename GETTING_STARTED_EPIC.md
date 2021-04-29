# Getting Started with PySlowFast

This document provides a brief intro of launching jobs in PySlowFast for training and testing. Before launching any job, make sure you have properly installed the PySlowFast following the instruction in [README.md](README.md) and you have prepared the dataset following [DATASET.md](slowfast/datasets/DATASET.md) with the correct format.

## Dataset

1.  AVA Dataset [official site](https://research.google.com/ava/download.html#ava_actions_download)
    * Processing [dataset and structure](https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md)
2.  EPIC Dataset [official site](https://epic-kitchens.github.io/2021#downloads) and [torrent](https://academictorrents.com/browse.php?search=EPIC&c6=1)
    * Processing [dataset and structure](https://github.com/epic-kitchens/epic-kitchens-slowfast)
## Configure
Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/home/rahul/workspace/epic/SlowFast:$PYTHONPATH
```
## Run

``` 
python tools\run_net.py --cfg path/to/<pretrained_model_config_file>.yaml
```

---

## A. Train a Standard Model from Scratch (AVA)

Here we can start with training a simple SLOWFAST models on AVA dataset:
- Configurations are in yaml configs file. Eg: (config/SLOWFAST_32x2_R50_SHORT.yaml)
  ```
    DATA:
      PATH_TO_DATA_DIR: path_to_your_dataset
  ```
- We can configure in the file or pass in the command line
  ``` python tools/run_net.py --cfg configs/AVA/SLOWFAST_32x2_R50_SHORT_V0.yaml DATA.PATH_TO_DATA_DIR /media/rahul/DTB2/data/AVA ```
- You may also want to add multiple other configurations
  ``` DATA_LOADER.NUM_WORKERS 0 NUM_GPUS 2 TRAIN.BATCH_SIZE 16 ```

If you want to launch a quick job for debugging on your local machine.

#### Resume from an Existing Checkpoint

- If your checkpoint is trained by PyTorch, then you can add the following line in the command line, or you can also add it in the YAML config:
  ``` TRAIN.CHECKPOINT_FILE_PATH path_to_your_PyTorch_checkpoint ```
- If the checkpoint in trained by Caffe2, then you can do the following:
  ``` TRAIN.CHECKPOINT_FILE_PATH path_to_your_Caffe2_checkpoint TRAIN.CHECKPOINT_TYPE caffe2 ```
- If you need to performance inflation on the checkpoint, remember to set `TRAIN.CHECKPOINT_INFLATE` to True.


### Test

We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job. 
- If only testing is preferred, you can set the `TRAIN.ENABLE` to False, 
- Train or download from [Model Zoo](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md#ava)
- Pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH. Eg: [SLOWFAST_32x2_R101_50_50_v2.1.pkl](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50_v2.1.pkl)
  ```
  python tools/run_net.py --cfg configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1_V0.yaml DATA.PATH_TO_DATA_DIR /media/rahul/DTB2/data/AVA \ 
        TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH output/SLOWFAST_32x2_R101_50_50_v2.1.pkl
  ```
---


## B. Train a Standard Model from Scratch (EPIC)
