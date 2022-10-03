import logging
import sys
import numpy as np

import tvm
from tvm import te
import topi
from topi.testing import conv2d_nchw_python

from tvm import autotvm

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
data = te.placeholder((N, CI, H, W), name='data')
kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype='float32')
#cfg = autotvm.get_config()
task = autotvm.task.create("conv2d_nchw.cuda",
                           args=([conv],),
                           target='cuda')
print(task.config_space)
