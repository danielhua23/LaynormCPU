import torch
#import my_cpu_extension #在UT里面直接这样call，对user体验不好，所以很多时候需要把这个pytorch cpp extension在python层面包装一层API
from lesson10.nn.layernorm import LayerNorm
x = torch.ones((16, 16), dtype=torch.float)
gamma = torch.ones((16), dtype=torch.float)
beta = torch.ones((16), dtype=torch.float) #要为16的倍数才行，不然avx512处理不了，得添加不能整除时候得case条件

ln = LayerNorm()
result = ln(x, gamma, beta)
print(result)