import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
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


class Model(nn.Module):
    def __init__(self, input_size, g):
        super(Model, self).__init__()

        self.g = g
        self.N = input_size

        # Defining some parameters
        w = torch.empty(input_size * input_size, 1)
        self.weights = nn.Parameter(nn.init.xavier_normal_(w))
    def forward(self, dx):
        self.g.requires_grad = False
        w = torch.matmul(self.g, self.weights ** 2) 
        # w = torch.matmul(g, weights ** 2) 
        
        f = w.reshape(self.N, self.N) # add exponential term
        ef = torch.softmax(-f, -1)
        z = ef @ dx
        z = z.sum(-1)
        return z, w



def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, T = X.shape   
    
    model = Model(N, g)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()

    # Generate data 
    input, target, _, _ = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        losses = 0
        for idx in tqdm(loader): 
            y = target[idx,]
            dx = input[idx, ]
            pred, _ = model(dx)
            pred = pred.squeeze(-1)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}"
        print(msg)

        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)


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


if __name__ == "__main__":

    sample_path = 'data/sample.pickle'
    data_path = 'data/sim.npy'
    model_path = 'model/sim.pt'
    

    train_size = 300
    batch_size = 50
    epochs = 50
    lr = 1.0
    
    p = 1


    X = np.load(data_path)
    _, d, _ = load_pickle(sample_path)
    X = torch.from_numpy(X).float()
         
    X_train = X[:, :train_size]

    shape = 'monotone_inc'

    if sys.argv[1] == 'train':
        train(X_train, d, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        collector = {'h': [], 'loss': []}
        hs = [1, 5] + list(range(10, 50, 10))
        for h in hs:
            print("h = ", h)
            forecast_path = f'output/sim_h{h}.pickle' if h in hs[:2] else None
            loss = forecast(X, d, p, train_size, h, model_path, forecast_path, shape, device='cpu')
            collector['h'].append(h)
            collector['loss'].append(loss)

        df = pd.DataFrame.from_dict(collector, orient='index')
        df.to_csv('output/sim.csv')






