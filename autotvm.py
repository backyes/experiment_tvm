import logging
import sys
import numpy as np

import tvm
from tvm import te
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python

from tvm import autotvm

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
data = te.placeholder((N, CI, H, W), name='data')
kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
conv = topi.nn.conv2d_nchw(data, kernel, strides, padding, dilation=1, out_dtype='float32')
#cfg = autotvm.get_config()
task = autotvm.task.create("conv2d_nchw",
                           args=(data, kernel, strides, padding, 1, 'float32'),
                           target='llvm')
print(task.config_space)

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
)

tuner = autotvm.tuner.XGBTuner(task)
tuner.tune(n_trial=20,
           measure_option=measure_option,
           callbacks=[autotvm.callback.log_to_file('conv2d.log')])
