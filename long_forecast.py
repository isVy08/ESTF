import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


# STATONARY
def long_forecast(input, model, h, T):
    preds = []
    t = 0
    while t < T:
        x = input[t: t+1]
        end = min(t+h, T)
        for i in range(t, end):
            x, _ = model(x)
            preds.append(x)
            x = x.unsqueeze(-1)
        t = i + 1
    return torch.cat(preds)


def forecast(X, d, p, train_size, h,
            model_path, forecast_path, 
            shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = X.shape 

    model = Model(N, g)
    load_model(model, None, model_path, device)
    model.eval()

    input, _, _, _ = generate_data(X, p)
    
    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    pred, F = model(input)
    if p == 1 and h > 1:
        pred = long_forecast(input, model, h, T)
    
    pred = torch.t(pred)
    out = torch.cat((X[:, :p], pred), dim=-1) 
    loss = loss_fn(out[:, train_size:T], X[:, train_size:T])
    print(loss.item())
    
    if forecast_path:
        F = F[:, 0].detach().numpy()
        out = out.detach().numpy()
        write_pickle([X, out, F], forecast_path)
    return loss.item()

## NON STATIONARY + ADJUST FOR MULTI-STEP FORECASTING
def long_forecast(input, input_indices, model, h, T, path):
    file = open(path, 'w+')
    t = 0
    pbar = tqdm(total = T + 2)
    while t < T:
        x = input[t: t+1]
        x_i = input_indices[t: t+1]
        end = min(t+h, T)
        for i in range(t, end):
            x, _ = model(x, x_i)
            file.write(str(x[0,].tolist())+'\n')
            x = x.unsqueeze(-1)
            x_i = input_indices[i: i+1]  
        pbar.update(t)      
        t = i + 1

    file.close()
    pbar.close()
    

def forecast(X, d, p, train_size, h,
            model_path, forecast_path, 
            shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    N, T = X.shape 

    input, target, input_indices, target_indices = generate_data(X, p)
    
    model = Model(N, T, g)
    load_model(model, None, model_path, device)
    model.eval()

    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    pred, F = model(input, input_indices)
    
    if p == 1 and h > 1:
        path = 'output/temp'
        long_forecast(input, input_indices, model, h, T, path)
        pred = [eval(i) for i in load(path)]
        pred = torch.Tensor(pred)

    pred = torch.t(pred)
    out = torch.cat((X[:, :p], pred), dim=-1) 
    loss = loss_fn(out[:, train_size:T], X[:, train_size:T])
    print(loss.item())

    if forecast_path:
        X = X.detach().numpy()
        F = F[:, 0].detach().numpy()
        out = out.detach().numpy()
        write_pickle([X, out, F], forecast_path)
    return loss.item()
