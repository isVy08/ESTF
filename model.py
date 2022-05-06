import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, N, T, gain=0.0075):
        
        super(Model, self).__init__()

        self.N = N
        self.T = T

        # Defining some parameters        
        w = torch.empty(N * N, T)       
        self.weights = nn.Parameter(nn.init.xavier_uniform_(w, gain=gain)) # nst sim: 0.008

    def forward(self, x, x_i, g):
        """
        x : [b, N, p]
        x_i : [b, p]
        """
    
        # Shape function
        F = torch.matmul(g, self.weights ** 2) # [N ** 2, T]
        
        f = F.t().reshape(-1, self.N, self.N) #[T, N, N]
        f = torch.softmax(f, -1) # [T, N, N]
        f_ = f[x_i]
        x_ = torch.swapaxes(x, 1, 2).unsqueeze(-1)

        Z = torch.matmul(f_, x_)
        Z = Z.sum((1, -1))
        return Z, F

