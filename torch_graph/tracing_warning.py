import torch

# https://stackoverflow.com/questions/66746307/torch-jit-trace-tracerwarning-converting-a-tensor-to-a-python-boolean-might-

class Foo(torch.nn.Module):
    def forward(self, tensor):
        # It is data dependent
        # Trace will only work with one path
        if tensor.max() > 0.5:
            return tensor ** 2
        return tensor


model = Foo()
model.eval()
traced = torch.jit.script(model) # No warnings
print(traced.code)
traced = torch.jit.trace(model, torch.randn(10)) # Warning
print(traced.code)


#  (base) shwangyanfei@wangyanfeideMBP torch_graph % python3 tracing_warning.py
#  def forward(self,
#      tensor: Tensor) -> Tensor:
#    _0 = bool(torch.gt(torch.max(tensor), 0.5))
#    if _0:
#      _1 = torch.pow(tensor, 2)
#    else:
#      _1 = tensor
#    return _1
#  
#  /Users/shwangyanfei/work/experiment_tvm/torch_graph/tracing_warning.py:8: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
#    if tensor.max() > 0.5:
#  def forward(self,
#      tensor: Tensor) -> Tensor:
#    return torch.pow(tensor, 2)
#  
