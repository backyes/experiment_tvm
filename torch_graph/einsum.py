import torch
import numpy as np

# https://zhuanlan.zhihu.com/p/361209187

a = torch.arange(9).reshape(3, 3)
# i = 3
torch_ein_out = torch.einsum('ii->i', [a]).numpy()
torch_org_out = torch.diagonal(a, 0).numpy()


print(a)
print(torch_ein_out)
print(torch_org_out)

torch_ein_out = torch.einsum('ij->ji', [a]).numpy()
torch_org_out = torch.transpose(a, 0, 1).numpy()
print(a)
print(torch_ein_out)
print(torch_org_out)



a = torch.randn(2,3,4,5,6)
a = torch.randn(2,2,2,2,2)
# i = 7, j = 9
torch_ein_out = torch.einsum('...ij->...ji', [a]).numpy()
torch_org_out = a.permute(0, 1, 2, 4, 3).numpy()

print(a)
print(torch_ein_out)
