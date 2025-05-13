import torch
import my_cpu_extension
# pytorch style API
class LayerNorm(torch.nn.Module):
    def __init__(self, gamma=1.0, beta=0.0):
        super().__init__() 
        # self.gamma = gamma
        # self.beta = beta
        
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        output = my_cpu_extension.layernorm(x, gamma, beta)
        return output
    