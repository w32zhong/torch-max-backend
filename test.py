import torch
from torch_max_backend import max_backend

@torch.compile(backend=max_backend)
def simple_math(x, y):
    return x + y * 2

# Usage
a = torch.tensor([1.0, 2.0, 3.0]).to('cuda:0')
b = torch.tensor([4.0, 5.0, 6.0]).to('cuda:0')
print(simple_math(a, b))  # Accelerated execution
