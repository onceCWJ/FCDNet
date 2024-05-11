# FCDNet: Frequency-Guided Complementary Dependency Modeling for Multivariate Time-Series Forecasting
## Requirements
- Python 3.8.3
- see `requirements.txt`

## If you find this code useful, please cite:
@article{CHEN2024106385,
title = {FCDNet: Frequency-guided complementary dependency modeling for multivariate time-series forecasting},
journal = {Neural Networks},
pages = {106385},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106385},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024003095},
author = {Weijun Chen and Heyuan Wang and Ye Tian and Shijie Guan and Ning Liu}
}


## Data Preparation

#### TXT File
Download Solar-Energy datasets from [https://github.com/laiguokun/multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data). Put into the `data/{solar_AL}` folder.

#### NPZ File

Download PEMS03, PEMS04, PEMS07, PEMS08 datasets from [https://github.com/Davidham3/ASTGCN/tree/master/data). Put into the `data/{PEMS03,PEMS04,PEMS07,PEMS08}` folder.

## Split dataset

Run the following commands to generate train/validation/test dataset at `data/{solar_AL,PEMS03,PEMS04,PEMS07,PEMS08}/{train,val,test}.npz`.

```
python generate_data.py --dataset PEMS03 --train_rate 0.6 --val_rate 0.2

python generate_data.py --dataset PEMS04 --train_rate 0.6 --val_rate 0.2

python generate_data.py --dataset PEMS07 --train_rate 0.6 --val_rate 0.2

python generate_data.py --dataset PEMS08 --train_rate 0.6 --val_rate 0.2

python generate_data.py --dataset Solar_AL
```

## Train Commands

* Solar-Energy
```
# Use Solar-Energy dataset
python train.py --dataset_dir=data/solar_AL
```
* PEMS03
```
# Use PEMS03 dataset
python train.py --dataset_dir=data/PEMS03
```
* PEMS04
```
# Use PEMS04 dataset
python train.py --dataset_dir=data/PEMS04
```
* PEMS07
```
# Use PEMS07 dataset
python train.py --dataset_dir=data/PEMS07 
```
* PEMS08
```
# Use PEMS08 dataset
python train.py --dataset_dir=data/PEMS08
```
## Acknowledgement
The paper is accepted by Neural Networks in 2024.5.7, please refer to this website for the pre-proof version: (https://www.sciencedirect.com/science/article/pii/S0893608024003095) And the link to the formal version will be given soon.
