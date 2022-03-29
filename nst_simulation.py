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

def generate_alphas(lower, upper, AT=4000):
    lower = 0.01
    alphas = [[lower]] + [ [ (1 - (t-1)/AT) * lower + (t-1)/AT * upper ]  for t in range(2, AT+1)]
    return torch.Tensor(alphas)


class Model(nn.Module):
    def __init__(self, N, T, g):
        
        super(Model, self).__init__()

        self.g = g
        self.N = N
        self.T = T
        self.alphas = generate_alphas(0.01, 3)

        # Defining some parameters        
        self.weights = nn.Parameter(nn.init.uniform_(torch.empty(N * N, 1)))

    def forward(self, x, x_i):
        """
        x : [b, N, p]
        x_i : [b, p]
        y_i : [b]
        """
        self.g.requires_grad = False
    
        # Shape function
        F = torch.matmul(self.g, self.weights ** 2) # [N ** 2, 1]
        W = torch.matmul(self.alphas, -F.t()) # [T, N ** 2]
        
        wg = W.reshape(-1, self.N, self.N) #[T, N, N]
        f = torch.softmax(wg, -1) # [T, N, N]
        
        Z = 0
        for p in range(x.size(-1)):
            steps = x_i[:, p]
        
            f_ = f[steps, :]

            a = (x[:, :, p]).unsqueeze(-1)
            z = torch.matmul(f_, a)
            Z += z.squeeze(-1)
        
        return Z, F



def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]

    F = np.log(d+1)
    F = torch.from_numpy(F)
    
    
    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T], alphas: [T]
    input, target, input_indices, target_indices = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    #  Intialize model
    model = Model(N, T, g)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()
    prev_loss = 1e+10

    for epoch in range(1, epochs + 1):
        train_losses = 0
        val_losses = 0
        for idx in tqdm(loader): 
            y = target[idx, ]
            x = input[idx, ]
            x_i = input_indices[idx, ]

            pred, F_hat = model(x, x_i)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            train_losses += loss.item()
            val_losses += loss_fn(F_hat[:, 0], F).item()

        train_loss = train_losses / len(loader)
        val_loss = val_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}"
        print(msg)
        if val_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = val_loss
        elif val_loss > prev_loss: 
            print(f'Early stopping at epoch {epoch}')
            break

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
            # print(i)
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
    

if __name__ == "__main__":


    sample_path = 'data/sample.pickle'
    data_path = 'data/nst_sim_data.csv'

    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    X = torch.from_numpy(X).float()
    _, d, _ = load_pickle(sample_path)
    X = X[:, :2000]


    train_size = 1000
    batch_size = 1000
    epochs = 10000
    lr = 0.001
    p = 1

    model_path = 'model/nst_sim.pt'
    shape = 'monotone_inc'


    if sys.argv[1] == 'train':
        X_train = X[:, :train_size]
        train(X_train, d, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        collector = {'h': [], 'loss': []}
        hs = [1, 10] + list(range(50, 250, 50))
        for h in hs:
            print("h = ", h)
            forecast_path = f'output/nst_sim_h{h}.pickle' if h in hs[:2] else None
            loss = forecast(X, d, p, train_size, h, model_path, forecast_path, shape, device='cpu')
            collector['h'].append(h)
            collector['loss'].append(loss)

        df = pd.DataFrame.from_dict(collector, orient='index')
        df.to_csv('output/nst_sim.csv')





