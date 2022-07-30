import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from model import Model
from torch.utils.data import DataLoader



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

def train(X, d, p, threshold, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape, q = threshold)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]
    if threshold is not None and threshold < 200:
        g = g.to_sparse()
    
    N, T = X.shape  

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T]
    input, target, input_indices, _ = generate_data(X, p)
    indices = list(range(T-p))
    loader = DataLoader(indices, batch_size=batch_size, shuffle=True)

    #  Intialize model
    model = Model(N, T, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()
    prev_loss = 1e+10

    
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
        
        train_loss = train_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}"
        print(msg)
        if train_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = train_loss

    



if __name__ == "__main__":

    import psutil, time
    process = psutil.Process(os.getpid())
    start = time.time()

    dataset = 'air'

    # Specify quantile value threshold
    threshold = None if sys.argv[2] == 'None' else int(sys.argv[2])

    sample_path = f'data/{dataset}/sample.pickle'
    data_path = f'data/{dataset}/data.npy'
    if threshold is None:
        model_path = f'model/{dataset}.pt'
        forecast_path = f'output/{dataset}.pickle'
    else:
        model_path = f'model/{dataset}_{threshold}.pt'
        forecast_path = f'output/{dataset}_{threshold}.pickle'
    

    train_size = 200
    batch_size = 50
    epochs = 100
    lr = 0.01
    
    p = 1


    X = np.load(data_path)
    _, d = load_pickle(sample_path)
    X, _  = normalize(X)
    X = torch.from_numpy(X).float()
         
    X_train = X[:, :train_size]

    shape = 'convex_dec'

    if sys.argv[1] == 'train':
        train(X_train, d, p, threshold, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        until = 165
        epochs = 100
        h = until
        from forecast import forecast, update
        forecast(X, d, p, threshold, train_size, lr, until, epochs, h, model_path, forecast_path, shape, device='cpu')
    
    end = time.time()
    print(f'Start: {time.ctime(start)}, End: {time.ctime(end)}')
    print(f'{threshold}: {process.memory_info().vms} - Computing time: {end - start} seconds')







