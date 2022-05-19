import os, sys
from utils import *
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


    sample_path = 'data/sample.pickle'

    data_path = sys.argv[1]
    forecast_path = sys.argv[2]
    model_path = sys.argv[3]
    

    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    X = torch.from_numpy(X).float()
    _, d = load_pickle(sample_path)
    
    train_size = 300
    batch_size = 300
    epochs = 100
    lr = 0.001
    p = 1

    shape = 'convex_dec'
    
    X_train = X[:, :train_size]

    threshold = None if sys.argv[4] == 'None' else int(sys.argv[4])

    train(X_train, d, p, threshold, model_path, batch_size, epochs, lr, shape, device='cpu')
    until = 200
    epochs = 100
    h = until
    from forecast import forecast, update
    forecast(X, d, p, threshold, train_size, lr, until, epochs, h, model_path, forecast_path, shape, device='cpu')