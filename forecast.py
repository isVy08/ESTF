import torch
from utils import *
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model import Model
from main import generate_data


def update(X_new, p, g, model, optimizer, loss_fn):
    x, y, x_i, _ = generate_data(X_new, p)
    y_hat, F = model(x, x_i, g)
    optimizer.zero_grad()
    loss = loss_fn(y_hat, y)
    loss.backward(retain_graph=True)        
    optimizer.step()
    return model, optimizer, F

    
def forecast(X, d, p, threshold, train_size, lr, until, epochs, h, 
            model_path, forecast_path, 
            shape, device):
    
    # if h = until < train_size: no-retraining

    g = basis_function(d, shape, q = threshold)
    g = torch.from_numpy(g).float()
    if threshold is not None and threshold < 200:
        g = g.to_sparse()
    
    N, T = X.shape[0], train_size 

    input, target, input_indices, _ = generate_data(X, p)
    
    model = Model(N, T, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    load_model(model, optimizer, model_path, device)
    loss_fn = nn.MSELoss()

    
    # Dynamic forecasting
    preds, F = model(input[:T-p, ], input_indices[:T-p, ], g) 
    preds = torch.cat((X.t()[:p, :], preds), dim=0)
    complete = False
    Fs = [F]


    
    while not complete:
        with torch.no_grad():   
            model.eval()         
            print(f'Forecasting the next {h} steps ...')
            
            # Forecast the next h < T steps, use the last h shape function estimates
            x = input[T - p: T + h - p,]
            hx = x.size(0)
            print('Forecasting size:', hx)
            x_i = torch.arange(train_size-hx, train_size).unsqueeze(-1)
            y_hat, _ = model(x, x_i, g)
            preds = torch.cat((preds, y_hat))
            L = preds.size(0)
            remaining = max(0, until + train_size - L)
            print(f'{remaining} steps until completion')
        
        if remaining == 0 :
            complete = True
            print('Finished !')
            break   

        T = L
        if not complete:
            with torch.enable_grad():  
                model.train()
                # Update model
                print('Updating model ...')
                for i in tqdm(range(epochs)):
                    X_new = preds[-train_size:, ].t()
                    model, optimizer, F = update(X_new, p, g, model, optimizer, loss_fn)
                
                Fs.append(F[:, -hx:])
        
          
    
    out = preds.t()
    
    T = X.shape[1]
    out = out[:, :T]
    
    loss = loss_fn(out, X)
    print(loss.item())

    F = torch.cat(Fs, 1)
    F = F[:, :T].detach().numpy()
    out = out.detach().numpy()  
    X = X.detach().numpy()
    write_pickle([X, out, F], forecast_path) 
