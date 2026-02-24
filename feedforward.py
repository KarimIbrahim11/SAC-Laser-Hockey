import torch
import numpy as np

torch.set_default_dtype(torch.float32)

_LOG_STD_MAX = 2
_LOG_STD_MIN = -20

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_fun=torch.nn.Tanh(), 
                 output_activation=None, use_layernorm=False):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.output_activation = output_activation
        self.activation_fun = activation_fun
        self.use_layernorm = use_layernorm
        
        layer_sizes = [self.input_size] + self.hidden_sizes
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(i, o) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ])
        
        if self.use_layernorm:
            self.layernorms = torch.nn.ModuleList([
                torch.nn.LayerNorm(o) for o in layer_sizes[1:]
            ])
        else:
            self.layernorms = None

        self.activations = torch.nn.ModuleList([activation_fun for _ in self.layers])
        self.readout = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if isinstance(self.activation_fun, torch.nn.ReLU):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                torch.nn.init.zeros_(m.bias)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_layernorm:
            for layer, layernorm, activation_fun in zip(self.layers, self.layernorms, self.activations):
                x = layer(x)
                x = layernorm(x)
                x = activation_fun(x)
        else:
            for layer, activation_fun in zip(self.layers, self.activations):
                x = activation_fun(layer(x))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def predict(self, x):
        with torch.no_grad():
            x_t = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            return self.forward(x_t).cpu().numpy()


class MultiHeadFeedforward(Feedforward):
    """
    Actor policy network.
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation_fun=torch.nn.ReLU(), use_layernorm=False):
        super().__init__(input_size=input_size, hidden_sizes=hidden_sizes[:-1], 
                         output_size=hidden_sizes[-1], activation_fun=activation_fun, 
                         output_activation=None, use_layernorm=use_layernorm)
        
        self.output_size = output_size
        
        if self.use_layernorm:
            self.ln_trunk_output = torch.nn.LayerNorm(self.hidden_sizes[-1])
        else:
            self.ln_trunk_output = None
        
        # Heads
        self.mu = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.log_sigma = torch.nn.Linear(self.hidden_sizes[-1], self.output_size)
        
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.zeros_(self.mu.bias)
        
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.constant_(self.log_sigma.bias, -0.5)

    def forward(self, x):
        if self.use_layernorm:
            for layer, layernorm, activation_fun in zip(self.layers, self.layernorms, self.activations):
                x = activation_fun(layernorm(layer(x)))
        else:
            for layer, activation_fun in zip(self.layers, self.activations):
                x = activation_fun(layer(x))
        
        x = self.readout(x)
        if self.use_layernorm:
            x = self.ln_trunk_output(x)
            
        x = self.activations[0](x)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        log_sigma = torch.clamp(log_sigma, min=_LOG_STD_MIN, max=_LOG_STD_MAX)
        return mu, log_sigma