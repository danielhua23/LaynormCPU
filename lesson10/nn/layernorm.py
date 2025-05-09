import torch
import my_cpu_extension

class LayerNorm(torch.nn.Module):
    def __init__(self, gamma=1.0, beta=0.0):
        super().__init__() # 初始化父类，不写会报AttributeError: 'LayerNorm' object has no attribute '_backward_hooks
        # self.gamma = gamma
        # self.beta = beta
        
    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor):
        output = my_cpu_extension.layernorm(x, gamma, beta)[0]
        return output
    