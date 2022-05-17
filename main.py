import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model import Model
from torch.utils.data import DataLoader

# Specify quantile value threshold
threshold = None if sys.argv[2] == 'None' else int(sys.argv[2])

def generate_data(X, p):
    
    input, target = [], []
    input_indices = []
    
    T = X.size(1)
    target_indices = torch.arange(p, T) 

    for i in range(p, T):        
        target.append(X[:, i])
        input.append(X[:, i-p:i])
        input_indices.append(torch.arange(i-p, i))
    
    return torch.stack(input), torch.stack(target), torch.stack(input_indices), target_indices

def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape, q = threshold)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]
    if threshold is not None and threshold < 200:
        g = g.to_sparse()
    
    N, T = X.shape  

    # Validation size 
    V = 50

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T]
    input, target, input_indices, _ = generate_data(X, p)
    indices = list(range(T-p-V))
    loader = DataLoader(indices, batch_size=batch_size, shuffle=True)

    #  Intialize model
    model = Model(N, T, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()
    tloss = vloss = 5.0

    
    for epoch in range(1, epochs + 1):
        train_losses = 0
        for idx in tqdm(loader): 
            y = target[idx, ]
            x = input[idx, ]
            x_i = input_indices[idx, ]

            pred, _ = model(x, x_i, g)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            train_losses += loss.item()
        
        # Validation
        pred, _ = model(input[-V:, ], input_indices[-V:, ], g)
        val_loss = loss_fn(pred, target[-V:, ])
        
        train_loss = train_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}"
        print(msg)
        if val_loss <= vloss and train_loss < tloss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            vloss = val_loss
            tloss = train_loss

def update(X_new, p, g, epochs, model, optimizer, loss_fn):
    for _ in tqdm(range(epochs)):
        x, y, x_i, _ = generate_data(X_new, p)
        y_hat, F = model(x, x_i, g)
        optimizer.zero_grad()
        loss = loss_fn(y_hat, y)
        loss.backward(retain_graph=True)        
        optimizer.step()
    return model, optimizer, F

    
def forecast(X, d, p, train_size, lr, until, epochs, h, 
            model_path, forecast_path, 
            shape, device):

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
    complete = False
    Fs = [F]
    

    while not complete:
        with torch.no_grad():   
            model.eval()         
            print(f'Forecasting the next {until} steps ...')
            t = 0
            while t < (train_size - p) and not complete:
                i = t + T
                # Apply long forecasting
                x = X.t()[i - p: i, :].reshape(1, N, -1)
                for _ in range(h):
                    x_i = torch.arange(t, t + p).unsqueeze(0)
                    y_hat, _ = model(x, x_i, g)
                    preds = torch.cat((preds, y_hat))
                    
                    L = preds.size(0)
                    remaining = max(0, until + train_size - L) 
                    print(f'{remaining} steps until completion')
                    x = preds[-p:, ].reshape(1, N, -1)
                    t += 1

                    if L >= until + train_size - p:
                        complete = True
                        print('Finished !')
                        break
        T = L
        if not complete:
            model.train()
            # Update model
            X_new = preds[-train_size:, ].t()
            X_new.requires_grad = True
            print('Updating model ...')
            model, optimizer, F = update(X_new, p, g, epochs, model, optimizer, loss_fn)
            Fs.append(F)
    
    
    out = preds.t()
    out = torch.cat((X[:, :p], out), dim=-1)
    T = out.shape[1]
    X = X[:, :T]
    
    loss = loss_fn(out, X)
    print(loss.item())

    F = torch.cat(Fs, 1)
    F = F[:, :T].detach().numpy()
    out = out.detach().numpy()  
    X = X.detach().numpy()
    write_pickle([X, out, F], forecast_path) 
    

if __name__ == "__main__":

    import psutil, time
    process = psutil.Process(os.getpid())
    start = time.time()


    sample_path = 'data/air/sample.pickle'
    data_path = 'data/air/data.npy'
    if threshold is None:
        model_path = f'model/air.pt'
        forecast_path = f'output/air.pickle'
    else:
        model_path = f'model/air_{threshold}.pt'
        forecast_path = f'output/air_{threshold}.pickle'
    

    train_size = 300
    batch_size = 50
    epochs = 100
    lr = 0.001
    
    p = 1


    X = np.load(data_path)
    _, d = load_pickle(sample_path)
    X = torch.from_numpy(X).float()
         
    X_train = X[:, :train_size]

    shape = 'convex_dec'

    if sys.argv[1] == 'train':
        train(X_train, d, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        until = 65
        epochs = 100
        h = 1
        forecast(X, d, p, train_size, lr, until, epochs, h, model_path, forecast_path, shape, device='cpu')
    
    end = time.time()
    print(f'Start: {time.ctime(start)}, End: {time.ctime(end)}')
    print(f'{threshold}: {process.memory_info().vms} - Computing time: {end - start} seconds')







