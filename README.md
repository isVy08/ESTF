# STVAR
This repo includes codes for reproducing the experiments in the paper Spatio-temporal Forecasting with Shape Functions

## Dependencies
STVAR requires Python 3.7+, R 3.6+ and the following packages

- `numpy`
- `torch`
- `pandas`
- `tqdm`
-  `psutil`
-  `tensorflow`

Create the following directories
```
mkdir data output model
mkdir model/stationary model/non_stationary
mkdir output/stationary output/non_stationary
```

## Data
Download the data zip files from this [Google Drive link](https://drive.google.com/drive/folders/1I_nSpdvV7zx8-Sv4eLVJ8X5RFnSa-9_p?usp=sharing) and extract them in `data/`. 
* `stationary.zip` and `non_stationary.zip` respectively contains 100 csv files used in our simulation. 
Location indices and distances are the same for both experiments, which can be retrieved from `sample.pickle`. 
We use `R` to create simulation data. If you want to creat one from sratch, refer to `r-script.ipynb`. 

* `air.zip` contains the original dataset and the processed resources for the real-case experiment on Air Quality data. 
`air/data.npy` is a `Numpy` data object directly used to train our model. The corresponding location indices and distances are stored in `air/sample.pickle`.
`data_prep.py` details how to generate this data. 

## Simulation
To run the stationary and non_stationary simulation on 100 datasets, 
```
bash run_st.sh
bash run_nst.sh
```
The last argument specifies the quantile threshold value to compute basis functions. If `threshold = None`, the computation is based on ordered statistics. 
The scripts perform training and forecasting altogether. 

## Real case
To train the model, for example with `quantile = XX` , run 
```
python air.py train XX
```
After training, the model will be automatically saved in `model/air_XX.pt`. To do forecasting, 
```
python air.py val XX
```
## Baselines
In this repo, we provide scripts to run the deep learning baseline models in their respective folders. 
The codes are gratefully adapted from [DC-RNN repo](https://github.com/liyaguang/DCRNN), [FC-GAGA repo](https://github.com/boreshkinai/fc-gaga), 
[GMAN repo](https://github.com/zhengchuanpan/GMAN) and [ConvLSTM repo](https://github.com/giserh/ConvLSTM-2).

These models require input data in `.h5` format. `air/data.h5` is an equivalent version for Air quality data, which can also be reproduced from `data_prep.py`. 
To generate `.h5` files for simulation and other utils, run 
```
python data_generator.py data/stationary/ 123
python data_generator.py data/non_stationary/ 123
```
## Citation
If you use the codes or datasets in this repository, please cite our paper.


