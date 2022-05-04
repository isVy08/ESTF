import torch
import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, N, T):
        
        super(Model1, self).__init__()

        self.N = N
        self.T = T

        # Defining some parameters        
        w = torch.empty(N * N, T)       
        self.weights = nn.Parameter(nn.init.xavier_normal_(w)) # nst sim: 0.008

    def forward(self, x, x_i, g):
        """
        x : [b, N, p]
        x_i : [b, p]
        """
    
        # Shape function
        F = torch.matmul(g, self.weights ** 2) # [N ** 2, T]
        
        f = F.t().reshape(-1, self.N, self.N) #[T, N, N]
        f = torch.softmax(f, -1) # [T, N, N], comment this out for nst sim
        f_ = f[x_i]
        x_ = torch.swapaxes(x, 1, 2).unsqueeze(-1)

        Z = torch.matmul(f_, x_)
        Z = Z.sum((1, -1))
        return Z, F

class Model2(nn.Module):
    def __init__(self, N, T):
        
        super(Model2, self).__init__()

        self.N = N
        self.T = T

        # Defining some parameters        
        self.weights = nn.Parameter(nn.init.normal_(torch.empty(N * N, 1)))
        self.alphas = nn.Parameter(nn.init.uniform_(torch.empty(1, T)))


    def forward(self, x, x_i, g):
        """
        x : [b, N, p]
        x_i : [b, p]
        """
        # Shape function
        F = torch.matmul(g, self.weights ** 2) # [N ** 2, 1]
        F = torch.matmul(F, self.alphas)
        
        wg = F.t().reshape(-1, self.N, self.N) #[T, N, N]
        # f = torch.softmax(-wg, -1) # [T, N, N]
        f = wg
        f_ = f[x_i]
        x_ = torch.swapaxes(x, 1, 2).unsqueeze(-1)

        Z = torch.matmul(f_, x_)
        Z = Z.sum((1, -1))
        return Z, F