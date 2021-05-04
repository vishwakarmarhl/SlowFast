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

### Test

We have `TRAIN.ENABLE` and `TEST.ENABLE` to control whether training or testing is required for the current job. 
- If only testing is preferred, you can set the `TRAIN.ENABLE` to False, 
- Train or download from [Model Zoo](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md#ava)
- Pass the path to the model you want to test to TEST.CHECKPOINT_FILE_PATH. Eg: [SLOWFAST_32x2_R101_50_50_v2.1.pkl](https://dl.fbaipublicfiles.com/pyslowfast/model_zoo/ava/SLOWFAST_32x2_R101_50_50_v2.1.pkl)
  ```
  python tools/run_net.py --cfg configs/AVA/c2/SLOWFAST_32x2_R101_50_50_v2.1_V0.yaml DATA.PATH_TO_DATA_DIR /media/rahul/DTB2/data/AVA TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH output/SLOWFAST_32x2_R101_50_50_v2.1.pkl
  ```
---


## B. Train a Standard Model from Scratch (EPIC)

```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_V0.yaml NUM_GPUS 1 OUTPUT_DIR ./output EPICKITCHENS.VISUAL_DATA_DIR /media/rahul/DTB2/data/epic/EPIC-KITCHENS  EPICKITCHENS.ANNOTATIONS_DIR /media/rahul/DTB2/data/epic/epic-kitchens-100-annotations EPICKITCHENS.DATA_LOAD_SELECTOR_LIST [\"P01\"] 
```
- sample log
    ``` [05/03 12:00:05][INFO] train_net.py: 568: Start epoch: 8
    [05/03 12:00:12][INFO] logging.py:  99: json_stats: {"_type": "train_iter", "dt_data": 0.01517, "dt_diff": 0.55199, "dt_net": 0.53681, "epoch": "8/30", "eta": "1:12:41", "gpu_mem": "5.22G", "iter": "10/344", "loss": 1.63013, "lr": 0.01000, "mem": 6, "noun_loss": 1.70481, "noun_top1_acc": 56.25000, "noun_top5_acc": 84.37500, "top1_acc": 37.50000, "top5_acc": 81.25000, "verb_loss": 1.72457, "verb_top1_acc": 43.75000, "verb_top5_acc": 90.62500}
    ``` 

### Test (EPIC)

```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_V0.yaml NUM_GPUS 1 OUTPUT_DIR ./output EPICKITCHENS.VISUAL_DATA_DIR /media/rahul/DTB2/data/epic/EPIC-KITCHENS  EPICKITCHENS.ANNOTATIONS_DIR /media/rahul/DTB2/data/epic/epic-kitchens-100-annotations TRAIN.ENABLE False TEST.ENABLE True TEST.CHECKPOINT_FILE_PATH ./output/SlowFast.pyth 
```
- sample log
    ``` [05/03 15:28:29][INFO] logging.py:  99: json_stats: {"cur_iter": "554", "eta": "0:00:00", "split": "test_iter", "time_diff": 0.02796}
    [05/03 15:28:29][INFO] logging.py:  99: json_stats: {"noun_top1_acc": "55.82", "noun_top5_acc": "80.23", "split": "test_final", "verb_top1_acc": "62.71", "verb_top5_acc": "92.88"}
    [05/03 15:28:29][INFO] test_net.py: 178: Successfully saved prediction results to ./output/testrun
    [05/03 15:28:29][INFO] logging.py:  99: json_stats: {"noun_top1_acc": "55.82", "noun_top5_acc": "80.23", "split": "test_final", "verb_top1_acc": "62.71", "verb_top5_acc": "92.88"}
    ```