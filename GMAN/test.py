import torch
import math
import time, os
import numpy as np
from tqdm import tqdm
from utils_ import log_string, metric
from utils_ import load_data

def test(args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, fullX, fullTE, fullY, SE) = load_data(args)
    num_samples, _, num_vertex = fullX.shape
    num_batch = math.ceil(num_samples / args.batch_size)
 
    model = torch.load(args.model_file)

    # test model
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    model = torch.load(args.model_file)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():

        Pred = []
        for batch_idx in tqdm(range(num_batch)):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_samples, (batch_idx + 1) * args.batch_size)
            X = fullX[start_idx: end_idx]
            TE = fullTE[start_idx: end_idx]
            pred_batch = model(X, TE)
            Pred.append(pred_batch.detach().clone())
            del X, TE, pred_batch

        Pred = torch.from_numpy(np.concatenate(Pred, axis=0))
        
    test_mae, test_rmse, test_mape = metric(Pred, fullY)
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae, test_rmse, test_mape * 100))
    log_string(log, 'performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []
    print(Pred.shape)
    for step in range(args.num_pred):
        mae, rmse, mape = metric(Pred[:, step], fullY[:, step])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        log_string(log, 'step: %02d         %.2f\t\t%.2f\t\t%.2f%%' %
                   (step + 1, mae, rmse, mape * 100))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    log_string(
        log, 'average:         %.2f\t\t%.2f\t\t%.2f%%' %
             (average_mae, average_rmse, average_mape * 100))
    print('Saving predictions')
    path = args.traffic_file.replace('stvar.h5','full_predictions.npz')
    np.savez_compressed(
    path,
    input=fullX,
    truth=fullY,
    prediction=Pred
    
    )
