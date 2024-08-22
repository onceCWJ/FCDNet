# FCDNet: Frequency-Guided Complementary Dependency Modeling for Multivariate Time-Series Forecasting
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

# Citation Request
If you find this codebase helpful for your research, please consider citing the following paper:

```plaintext
@article{chen2023fcdnet,
  title={FCDNet: Frequency-Guided Complementary Dependency Modeling for Multivariate Time-Series Forecasting},
  author={Chen, Weijun and Wang, Heyuan and Tian, Ye and Guan, Shijie and Liu, Ning},
  journal={arXiv preprint arXiv:2312.16450},
  year={2023}
}

Note: The reason for the voluntary withdrawal of our paper from Neural Networks, after it was accepted, was due to a dispute over the order of authors. This withdrawal is unrelated to the content of the paper itself.


