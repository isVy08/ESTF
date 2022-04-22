import os, sys
from utils import *
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


class Square(nn.Module):
    def forward(self, x):
        return torch.square(x)

class Model(nn.Module):
    def __init__(self, N, gain):
        
        super(Model, self).__init__()

        self.N = N
        # Defining some parameters 
        w = torch.empty(N * N, 1)       
        self.weights = nn.Parameter(nn.init.xavier_normal_(w, gain=gain))

    def forward(self, x, g):
        """
        x : [N, p]
        """
        # Shape function
        F = torch.matmul(g, self.weights ** 2) # [N ** 2, 1]
        # F = self.layer(g)
        
        wg = F.reshape(self.N, self.N) #[N, N]
        f = torch.softmax(-wg, -1) # [N, N]
        Z = f @ x
        Z = Z.sum(-1)
        return Z, F



def train(X, d, p, path, epochs, lr, shape, F, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float() # [N ** 2, N ** 2]

    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1], target: [T - 1, N], input_indices: [T-1, p], target_indices: [T], alphas: [T]
    input, target, input_indices, target_indices = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    F_est = torch.zeros_like(F)
    loss_fn = nn.MSELoss()

    for i in range(T-1):
        print("Training ")
        x = input[i, ]
        y = target[i, ]
        if i < 50:
            gain = 0.50
        else:
            gain = 0.75
        model = Model(N, gain)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        prev_loss = 1e+10
        for epoch in range(1, epochs+1):
            pred, F_hat = model(x, g)
            optimizer.zero_grad()
            loss = loss_fn(pred, y) # + loss_fn(F_hat, F[:, i:i+1])
            loss.backward()
            
            optimizer.step()
            
            vloss = loss_fn(F_hat, F[:, i:i+1]).item()

            msg = f"Epoch: {epoch}, Train loss: {loss:.5f}, Val loss: {vloss:.5f}"
            print(msg)
            if vloss < prev_loss:
                print(f'Continue with i = {i}  ...')
                prev_loss = vloss
                F_est[:, i] = F_hat.squeeze(-1)
            elif vloss > prev_loss:
                print('-' * 50)
                break

    
    # Save forecasting
    F_hat = F_est.detach().numpy().transpose()
    np.save(path, F_hat)


if __name__ == "__main__":

    sample_path = 'data/sample.pickle'

    i = int(sys.argv[1])
    # forecast_path = sys.argv[2]
    # model_path = sys.argv[3]

    data_path = f'data/nst_sim/csv/s{i}.csv'
    path = f'output/nst_sim/Fhat{i}.npy'


    df = pd.read_csv(data_path)
    X = df.iloc[:, 1:].to_numpy()

    X = torch.from_numpy(X).float()
    _, d = load_pickle(sample_path)
    

    train_size = 302
    batch_size = 10
    epochs = 100
    lr = 0.01
    p = 1

    
    shape = 'monotone_inc'
    F = np.load('data/nst_sim/F.npy')
    F = torch.from_numpy(F[i, :train_size, :].transpose()).float()


    X_train = X[:, :train_size]
    train(X_train, d, p, path, epochs, lr, shape, F, device='cpu')
 




