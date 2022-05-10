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

def train(X, d, p, model_path, batch_size, epochs, lr, shape, F, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]

    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T], alphas: [T]
    input, target, input_indices, _ = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    #  Intialize model
    model = Model(N, T, 0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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

            pred, F_hat = model(x, x_i, g)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            train_losses += loss.item()
            val_losses += loss_fn(F_hat, F).item()

        train_loss = train_losses / len(loader)
        val_loss = val_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}"
        print(msg)
        if train_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = train_loss


def update(X_new, p, g, epochs, model, optimizer, loss_fn):
    for _ in tqdm(range(epochs)):
        x, y, x_i, _ = generate_data(X_new, p)
        y_hat, F = model(x, x_i, g)
        optimizer.zero_grad()
        loss = loss_fn(y_hat, y)
        loss.backward(retain_graph=True)        
        optimizer.step()
    return model, optimizer, F

def forecast(X, d, p, train_size, lr, until, epochs, 
            model_path, forecast_path, 
            shape, device):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    N, T = X.shape[0], train_size 

    input, target, input_indices, _ = generate_data(X, p)
    
    model = Model(N, T, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    load_model(model, optimizer, model_path, device)
    loss_fn = nn.MSELoss()

    
    # Dynamic forecasting
    preds, F = model(input[:T+1-p, ], input_indices[:T+1-p, ], g) 
    complete = False
    Fs = [F]
    while not complete:
        with torch.no_grad():            
            print(f'Forecasting the next {train_size} steps ...')
            for t in range(train_size):
                i = t + T
                x = X.t()[i - p: i, :].reshape(1, N, -1)
                x_i = torch.LongTensor([[t]])
                y_hat, _ = model(x, x_i, g)
                preds = torch.cat((preds, y_hat))

                L = preds.size(0)
                remaining = max(0, until + train_size - L) 
                print(f'{remaining} steps until completion')
        
                if L >= until + train_size - p:
                    complete = True
                    break
            
            Fs.append(F)
        T = L
        if not complete:
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
 
    out = out.detach().numpy()  
    
    F = torch.cat(Fs, 1)
    F = F[:, :T].detach().numpy()
    X = X.detach().numpy()
    write_pickle([X, out, F], forecast_path) 

if __name__ == "__main__":


    sample_path = 'data/sample.pickle'

    data_path = sys.argv[1]
    forecast_path = sys.argv[2]
    model_path = sys.argv[3]
    F_path = sys.argv[4]
    i = int(sys.argv[5])

    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    X = torch.from_numpy(X).float()
    _, d = load_pickle(sample_path)


    train_size = 300
    batch_size = 50
    epochs = 100
    lr = 1e-3
    p = 1

    F = np.load(F_path)
    F = torch.from_numpy(F[i, :train_size, :].transpose()).float()


    shape = 'convex_dec'
    
    X_train = X[:, :train_size]


    train(X_train, d, p, model_path, batch_size, epochs, lr, shape, F, device='cpu')
    until = 200
    forecast(X, d, p, train_size, lr, until, 100, model_path, forecast_path, shape, device='cpu')




