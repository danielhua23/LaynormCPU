import torch
#import my_cpu_extension #在UT里面直接这样call，对user体验不好，所以很多时候需要把这个pytorch cpp extension在python层面包装一层API
from lesson10.nn.layernorm import LayerNorm
x = torch.ones((5, 5), dtype=torch.float)
gamma = torch.ones((1), dtype=torch.float)
beta = torch.tensor([0], dtype=torch.float)

ln = LayerNorm()
result = ln(x, gamma, beta)
print(result)