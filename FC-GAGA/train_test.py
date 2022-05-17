import sys, os
import tensorflow as tf
import numpy as np
print("Tensorflow version:", tf.__version__)
from dataset import Dataset
from model import Trainer, Parameters
from model import hyperparams_defaults as hyperparams_dict


dataname = sys.argv[2]
LOGDIR = f"./logs/{dataname}_train"
DATADIR = f"./data"

import os, psutil, time
process = psutil.Process(os.getpid())
start = time.time()

def insert_dict(d, k, v):
    previous = d.get(k, [])
    d[k] = previous + [v]
    return d

'''
Default params in model.py : 
- epochs: 100
- lr: 0.001
- history_length = horizon = 1
'''

hyperparams_dict["dataset"] = dataname

if dataname  == 'sim':
    hyperparams_dict["init_learning_rate"] = 1e-2
    

print("*********************************")
print("FC-GAGA parameters:")
print(hyperparams_dict)
print("LOADING DATA")
print("*********************************")

dataset = Dataset(name=hyperparams_dict["dataset"], 
                  horizon=hyperparams_dict["horizon"], 
                  history_length=hyperparams_dict["history_length"],
                  path=DATADIR)

hyperparams_dict["num_nodes"] = dataset.num_nodes
hyperparams = Parameters(**hyperparams_dict)

print("*********************************")
print("TRAINING MODELS")
print("*********************************")


trainer = Trainer(hyperparams=hyperparams, logdir=LOGDIR)

if sys.argv[1] == 'train':
    trainer.fit(dataset=dataset)

    for i in range(len(trainer.models)):
        # Save models
        model = trainer.models[i].model
        model.save_weights(f'model/{dataname}-{i}.hdf5')


    print("*********************************")
    print("COMPUTING METRICS")
    print("*********************************")

    file = open(LOGDIR, 'w+')

    early_stop_mae_h_repeats = dict()
    early_stop_mape_h_repeats = dict()
    early_stop_rmse_h_repeats = dict()
    early_stop_mae_h_ave = dict()
    early_stop_mape_h_ave = dict()
    early_stop_rmse_h_ave = dict()
    for i, h in enumerate(trainer.history):
        early_stop_idx = np.argmin(h['mae_val'])
        early_stop_mae = np.round(h['mae_test'][early_stop_idx], decimals=3)
        msg = f"Early stop test error model {trainer.folder_names[i]}:, Avg MAE = {early_stop_mae}"
        file.write(msg + '\n')
        print(msg)
        for horizon in range(3, hyperparams.horizon+1, 3):

            early_stop_mae_h_repeats = insert_dict(early_stop_mae_h_repeats, k=horizon, v=h[f'mae_test_h{horizon}'][early_stop_idx])
            early_stop_mape_h_repeats = insert_dict(early_stop_mape_h_repeats, k=horizon, v=h[f'mape_test_h{horizon}'][early_stop_idx])
            early_stop_rmse_h_repeats = insert_dict(early_stop_rmse_h_repeats, k=horizon, v=h[f'rmse_test_h{horizon}'][early_stop_idx])
            
            msg = f"Horizon {horizon} MAE: {np.round(early_stop_mae_h_repeats[horizon][-1], decimals=2)}\n \
                Horizon {horizon} MAPE:, {np.round(early_stop_mape_h_repeats[horizon][-1], decimals=2)}\n \
                Horizon {horizon} RMSE: {np.round(early_stop_rmse_h_repeats[horizon][-1], decimals=2)}."
            
            print(msg)

        for horizon in range(3, hyperparams.horizon+1, 3):
            early_stop_mae_h_ave[horizon] = np.round(np.mean(early_stop_mae_h_repeats[horizon]), decimals=2)
            early_stop_mape_h_ave[horizon] = np.round(np.mean(early_stop_mape_h_repeats[horizon]), decimals=2)
            early_stop_rmse_h_ave[horizon] = np.round(np.mean(early_stop_rmse_h_repeats[horizon]), decimals=2)

    print()
    print("Average MAE:", early_stop_mae_h_ave)
    print("Average MAPE:", early_stop_mape_h_ave)
    print("Average RMSE:", early_stop_rmse_h_ave)
    file.close()

    print(process.memory_info().vms)  # in bytes 
    end = time.time()
    print(f'Computing time: {end - start} seconds')


# FULL PREDICTIONS

elif sys.argv[1] == 'val':

    i = sys.argv[3]

    from utils import MetricsCallback
    path = f'model/{dataname}-{i}.hdf5'
    metrics = MetricsCallback(dataset=dataset, logdir=LOGDIR)
    best_model = trainer.models[-1].model
    best_model.load_weights(path)
    predictions = best_model.predict({"history": metrics.full_data["x"][...,0], 
                                        "node_id": metrics.full_data["node_id"],
                                        "time_of_day": metrics.full_data["x"][...,0]})
    np.savez_compressed(
        os.path.join(sys.argv[4]),
        input=metrics.full_data["x"],
        truth=metrics.full_data["y"],
        prediction=predictions['targets']
        
        )