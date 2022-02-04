import torch
from utils import *
import numpy as np
from tqdm import tqdm
from main import moving_average_standardize
from torch.distributions import uniform


def generate_data(T, N, W, p, seed=8):

    torch.manual_seed(seed)
    
    distribution = uniform.Uniform(torch.Tensor([0.1]),torch.Tensor([0.3]))
    X = distribution.sample((N,))
    W = torch.from_numpy(W.reshape(1, N, N)).float()

    # predict delta_X
    pred_DX = torch.t(X)
    pred_X = []
    for _ in tqdm(range(p, T)):
        dx = pred_DX[-p:, :].unsqueeze(-1)
        z = W @ dx
        # z = moving_average_standardize(z, p)
        new_dx = z.sum(0).reshape(1,-1)
        pred_X.append(X[:, -1:] + torch.t(new_dx))
        pred_DX = torch.cat((pred_DX, new_dx), dim=0)


    pred_X = torch.cat(pred_X, dim=1)
    X = torch.cat((X, pred_X), dim=1)
    return X.detach().numpy()

if __name__ == "__main__":

    
    data = np.load('data/data_small.npy')
    N, T = data.shape
    sp, d, _ = load_pickle('sample_small.pickle')
    W = 1e-2 * np.log(d+1)
    p = 1
    X = generate_data(T, N, W, p)
    np.save('data/data_sim.npy', X) 
    print(X.shape, X.min(), X.max())



