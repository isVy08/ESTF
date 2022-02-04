import torch

class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(RNN, self).__init__()

        # Defining some parameters
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Defining the layers
        # RNN Layer
        self.rnn = torch.nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = torch.nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
        
        # Fsit into the fully connected layer
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.randn(self.n_layers, batch_size, self.hidden_size)
        c0 = torch.randn(self.n_layers, batch_size, self.hidden_size)
        return (h0, c0)