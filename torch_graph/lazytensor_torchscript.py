import lazy_tensor_core as ltc
import torch

ltc._LAZYC._ltc_init_ts_backend()
device = torch.device('lazy')
dtype = torch.float32

x = torch.randn((4, 5), device=device, dtype=dtype)

# Capture lazy computation and convert to TorchScript IR
graph_str = ltc._LAZYC._get_ltc_tensors_backend([
    x + x
])

print(graph_str)
graph = torch._C.parse_ir(graph_str)
