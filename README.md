# FCDNet
## Requirements
- Python 3.8.3
- see `requirements.txt`

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
The copyright of the code belongs to the authors of the paper. Before the official publication of the paper, this code cannot be used in the research and commercial fields
