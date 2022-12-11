import torch
import torch.dynamo

def fn(a, b):
    x = a + b
    x = x / 2.0
    if x.sum() < 0:
        return x * -1.0
    return x
 
with torchdynamo.optimize(custom_compiler):   
   fn(torch.randn(10), torch.randn(10))
