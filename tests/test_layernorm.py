import torch
#import my_cpu_extension #在UT里面直接这样call，对user体验不好，所以很多时候需要把这个pytorch cpp extension在python层面包装一层API
from lesson10.nn.layernorm import LayerNorm
x = torch.ones((5, 5))
gamma = torch.ones((1))
beta = torch.tensor([0])

ln = LayerNorm()
# result = my_cpu_extension.layernorm_cpu(a, b)[0]
result = ln(x, gamma, beta)
print(result)