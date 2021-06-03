# Getting Started with PySlowFast

This document provides a brief intro of launching jobs in PySlowFast for training and testing. Before launching any job, make sure you have properly installed the PySlowFast following the instruction in [README.md](README.md) and you have prepared the dataset following [DATASET.md](slowfast/datasets/DATASET.md) with the correct format.

## Environment
- Change the environment.yml for PyTorch cudatoolkit other than [CUDA 11.0](https://pytorch.org/get-started/previous-versions/#v170) 
- Change Detectron2 dependency based on above CUDA [archive release](https://github.com/facebookresearch/detectron2/releases)
```
conda env create -n slowfast --file environment.yml --force
conda activate slowfast 
```

## Pretrained model

You can download our pretrained model on EPIC-KITCHENS-100 from [this link](https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=0)

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

## Train a Standard Model from Scratch (EPIC)

```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_V0.yaml NUM_GPUS 1 OUTPUT_DIR ./output EPICKITCHENS.VISUAL_DATA_DIR /media/rahul/DTB2/data/epic/EPIC-KITCHENS  EPICKITCHENS.ANNOTATIONS_DIR /media/rahul/DTB2/data/epic/epic-kitchens-100-annotations EPICKITCHENS.DATA_LOAD_SELECTOR_LIST [\"P01\"] 
```
- Sample log
    ``` 
    [05/03 12:00:05][INFO] train_net.py: 568: Start epoch: 8
    [05/03 12:00:12][INFO] logging.py:  99: json_stats: {"_type": "train_iter", "dt_data": 0.01517, "dt_diff": 0.55199, "dt_net": 0.53681, "epoch": "8/30", "eta": "1:12:41", "gpu_mem": "5.22G", "iter": "10/344", "loss": 1.63013, "lr": 0.01000, "mem": 6, "noun_loss": 1.70481, "noun_top1_acc": 56.25000, "noun_top5_acc": 84.37500, "top1_acc": 37.50000, "top5_acc": 81.25000, "verb_loss": 1.72457, "verb_top1_acc": 43.75000, "verb_top5_acc": 90.62500}
    ``` 

### Validate (EPIC)

```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_V0.yaml NUM_GPUS 1 OUTPUT_DIR ./output EPICKITCHENS.VISUAL_DATA_DIR /media/rahul/DTB2/data/epic/EPIC-KITCHENS  EPICKITCHENS.ANNOTATIONS_DIR /media/rahul/DTB2/data/epic/epic-kitchens-100-annotations TRAIN.ENABLE False TEST.ENABLE True TEST.CHECKPOINT_FILE_PATH ./output/SlowFast.pyth 
```
- Sample log
    ``` 
    [05/03 15:28:29][INFO] logging.py:  99: json_stats: {"cur_iter": "554", "eta": "0:00:00", "split": "test_iter", "time_diff": 0.02796}
    [05/03 15:28:29][INFO] logging.py:  99: json_stats: {"noun_top1_acc": "55.82", "noun_top5_acc": "80.23", "split": "test_final", "verb_top1_acc": "62.71", "verb_top5_acc": "92.88"}
    [05/03 15:28:29][INFO] test_net.py: 178: Successfully saved prediction results to ./output/testrun
    [05/03 15:28:29][INFO] logging.py:  99: json_stats: {"noun_top1_acc": "55.82", "noun_top5_acc": "80.23", "split": "test_final", "verb_top1_acc": "62.71", "verb_top5_acc": "92.88"}
    ```

After tuning the model's hyperparams using the validation set, we train the model that will be used for obtaining the test set's scores on the concatenation of the training and validation sets. To train the model on the concatenation of the training and validation sets run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/output_dir EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations EPICKITCHENS.TRAIN_PLUS_VAL True
```
To obtain scores on the test set (using the model trained on the concatenation of the training and validation sets) run:
```
python tools/run_net.py --cfg configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml NUM_GPUS num_gpus 
OUTPUT_DIR /path/to/experiment_dir EPICKITCHENS.VISUAL_DATA_DIR /path/to/dataset 
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations TRAIN.ENABLE False TEST.ENABLE True 
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth 
EPICKITCHENS.TEST_LIST EPIC_100_test_timestamps.pkl EPICKITCHENS.TEST_SPLIT test
```


## Dataset Preparation

- Please install all the requirements found in the original SlowFast repo ([link](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md))
* Add this repository to $PYTHONPATH.
```
export SLOWFASTPATH=/home/rahul/workspace/epic/epic-kitchens-slowfast
export PYTHONPATH=$SLOWFASTPATH:$PYTHONPATH
```
* From the annotation repository of EPIC-KITCHENS-100 ([link](https://github.com/epic-kitchens/epic-kitchens-100-annotations)), download: EPIC_100_train.pkl, EPIC_100_validation.pkl, and EPIC_100_test_timestamps.pkl. EPIC_100_train.pkl and EPIC_100_validation.pkl will be used for training/validation, while EPIC_100_test_timestamps.pkl will be used to obtain the scores to submit in the AR challenge.
* Download only the RGB frames of EPIC-KITCHENS-100 dataset using the download scripts found [here](https://github.com/epic-kitchens/epic-kitchens-download-scripts). 

The training/validation code expects the following folder structure for the dataset:
```
├── dataset_root
|   ├── P01
|   |   ├── rgb_frames
|   |   |   |    ├── P01_01
|   |   |   |    |    ├── frame_0000000000.jpg
|   |   |   |    |    ├── frame_0000000001.jpg
|   |   |   |    |    ├── .
|   |   |   |    .
|   ├── .
|   ├── P37
|   |   ├── rgb_frames
|   |   |   |    ├── P37_101
|   |   |   |    |    ├── frame_0000000000.jpg
|   |   |   |    |    ├── frame_0000000001.jpg
|   |   |   |    |    ├── .
|   |   |   |    .    
```
So, after downloading the dataset navigate under <participant_id>/rgb_frames for each participant and untar each video's frames in its corresponding folder, e.g for P01_01.tar you should create a folder P01_01 and extract the contents of the tar file inside.

- Ensure you have data extracted in the following path (command to extract multiple archives)
```
#!/bin/bash

#export EPIC_PATH=/media/rahul/DTB2/data/epic/EPIC-KITCHENS-100
export EPIC_PATH=/media/rahul/DTA2/data/epic/EPIC-KITCHENS
 
# Declare an array of participant_id
declare -a PartIdArray=("P01")
#declare -a PartIdArray=("P02" "P03" "P04" "P05" "P06" "P07" "P08" "P09" "P10")
#declare -a PartIdArray=("P11" "P12" "P13" "P14" "P15" "P16" "P17" "P18" "P19" "P20")
#declare -a PartIdArray=("P21" "P22" "P23" "P24" "P25" "P26" "P27" "P28" "P29" "P30")
#declare -a PartIdArray=("P31" "P32" "P33" "P34" "P35" "P36" "P37")

# Iterate the string array using for loop
for pid in ${PartIdArray[@]}; do
  cd $EPIC_PATH/$pid/rgb_frames
  echo "${pid} - Start Processing"
  for a in $(ls -1 *.tar); do 
    tar -xf $a --one-top-level ; 
    if [ $? -eq 0 ]; then
       echo "\t $a Success"
       #rm -f $a
    else
       echo "\t $a FAIL"
    fi
  echo "${pid} - End Processing"
  done
done


```


