import torch 

# https://zhuanlan.zhihu.com/p/493955209
 
 
def origin_func(x): 
    x = x**2 
    x = x**3 
    return x 
 
 
x = torch.rand(1, 2, 3, 4) 
jit_model = torch.jit.trace(origin_func, x) 
print(jit_model.graph) 
 

# graph(%x : Float(1, 2, 3, 4, strides=[24, 12, 4, 1], requires_grad=0, device=cpu)):
#   %1 : int = prim::Constant[value=2]() # /Users/shwangyanfei/opt/miniconda3/lib/python3.9/site-packages/torch/_tensor.py:32:0
#   %2 : Float(1, 2, 3, 4, strides=[24, 12, 4, 1], requires_grad=0, device=cpu) = aten::pow(%x, %1) # /Users/shwangyanfei/opt/miniconda3/lib/python3.9/site-packages/torch/_tensor.py:32:0
#   %3 : int = prim::Constant[value=3]() # /Users/shwangyanfei/opt/miniconda3/lib/python3.9/site-packages/torch/_tensor.py:32:0
#   %4 : Float(1, 2, 3, 4, strides=[24, 12, 4, 1], requires_grad=0, device=cpu) = aten::pow(%2, %3) # /Users/shwangyanfei/opt/miniconda3/lib/python3.9/site-packages/torch/_tensor.py:32:0
#   return (%4)

# 匹配用的子图定义，注意常量必须为[value=2]属性
pattern = """
    graph(%x):
        %const_2 = prim::Constant[value=2]()
        %out = aten::pow(%x, %const_2)
        return (%out)
"""

# 替换用的子图定义
replacement = """
    graph(%x):
        %out = aten::mul(%x, %x)
        return (%out)
"""


 # 使用刚才定义的 pattern与replacement来编辑graph
torch._C._jit_pass_custom_pattern_based_rewrite_graph(pattern, replacement,
                                                      jit_model.graph)

# 结果可视化，pow(x,2)被正确替换为mul(x,x)，pow(x,3)则保留原样不受影响。
print(jit_model.graph)
# graph(%x.1 : Float(1, 2, 3, 4, strides=[24, 12, 4, 1], requires_grad=0, device=cpu)):
#   %5 : Tensor = aten::mul(%x.1, %x.1)
#   %3 : int = prim::Constant[value=3]() # rewriter_test.py:7:0
#   %4 : Float(1, 2, 3, 4, strides=[24, 12, 4, 1], requires_grad=0, device=cpu) = aten::pow(%5, %3) # rewriter_test.py:7:0
#   return (%4)



