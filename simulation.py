import torch
from utils import *
import numpy as np
from tqdm import tqdm
from torch.distributions import uniform
from main import scale_distance


def generate_data(T, N, W, p):

    
    distribution = uniform.Uniform(torch.Tensor([0.0001]),torch.Tensor([0.0003]))
    X = distribution.sample((N,))

    # predict delta_X
    pred_X = [X]
    for _ in tqdm(range(p, T)):
        x = pred_X[-1]
        z = W @ x
        new_x = z.sum(-1).unsqueeze(-1)
        pred_X.append(new_x)
    
    # pred_X = [X]
    # for i in range(p, T):
    #     x = pred_X[-1] + pred_DX[i]
    #     pred_X.append(x)


    pred_X = torch.cat(pred_X, dim=1)
    return pred_X.detach().numpy()

if __name__ == "__main__":
    
    N = 30
    p = 1
    _, d, _ = load_pickle('data/sample_small.pickle')
    norm_d = scale_distance(d, None)
    norm_d = torch.from_numpy(norm_d.reshape(N, N)).float()
    W = 1e-2 * torch.log(norm_d+1)
    X = generate_data(100, N, W, p)
    print(X)

    np.save('data/data_sim.npy', X) 
    



