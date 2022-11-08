import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        x = x * 2
        x.add_(0)
        x = x.view(-1)
        if x[0] > 1:
            return x[0]
        else:
            return x[-1]


model = Model()

# seq of shape (batch_size, max_words)
x = torch.LongTensor([ 8, 29, 18,  1, 17,  3, 26,  6, 26,  5])

outputs = model(x)
print(outputs.shape)


model.eval()
with torch.no_grad():
	jit_model = torch.jit.trace(model, x)
#print(jit_model)
print(jit_model.code)
# def forward(self,
#     x: Tensor) -> Tensor:
#   x0 = torch.mul(x, CONSTANTS.c0)
#   x1 = torch.add_(x0, CONSTANTS.c1)
#   x2 = torch.view(x1, [-1])
#   return torch.select(x2, 0, 0)
print(jit_model.graph)
# graph(%self : __torch__.Model,
#       %x.1 : Long(10, strides=[1], requires_grad=0, device=cpu)):
#   %4 : Long(requires_grad=0, device=cpu) = prim::Constant[value={2}]() # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:9:0
#   %x.3 : Long(10, strides=[1], requires_grad=0, device=cpu) = aten::mul(%x.1, %4) # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:9:0
#   %6 : Long(requires_grad=0, device=cpu) = prim::Constant[value={0}]() # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:10:0
#   %7 : int = prim::Constant[value=1]() # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:10:0
#   %x.5 : Long(10, strides=[1], requires_grad=0, device=cpu) = aten::add_(%x.3, %6, %7) # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:10:0
#   %9 : int = prim::Constant[value=-1]() # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:11:0
#   %10 : int[] = prim::ListConstruct(%9)
#   %x : Long(10, strides=[1], requires_grad=0, device=cpu) = aten::view(%x.5, %10) # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:11:0
#   %17 : int = prim::Constant[value=0]() # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:13:0
#   %18 : int = prim::Constant[value=0]() # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:13:0
#   %19 : Long(requires_grad=0, device=cpu) = aten::select(%x, %17, %18) # /Users/shwangyanfei/work/experiment_tvm/torch_graph/example2.py:13:0
#   return (%19)

# 将模型序列化
jit_model.save('jit_model.pth')
# 加载序列化后的模型
jit_model = torch.jit.load('jit_model.pth')
