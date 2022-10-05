from __future__ import absolute_import, print_function


import tvm
import tvm.testing
from tvm import te
import numpy as np
m = te.var("m")
n = te.var("n")
X = te.placeholder((m, n), name="X")
s_state = te.placeholder((m, n))
s_init = te.compute((1, n), lambda _, i: X[0, i])
s_update = te.compute((m, n), lambda t, i: s_state[t - 1, i] + X[t, i])
s_scan = te.scan(s_init, s_update, s_state, inputs=[X])

s = te.create_schedule(s_scan.op)

print(tvm.lower(s, [X, s_scan], simple_mode=True))

num_thread = 256
block_x = te.thread_axis("blockIdx.x")
thread_x = te.thread_axis("threadIdx.x")
xo, xi = s[s_init].split(s_init.op.axis[1], factor=num_thread)
s[s_init].bind(xo, block_x)
s[s_init].bind(xi, thread_x)
xo, xi = s[s_update].split(s_update.op.axis[1], factor=num_thread)
s[s_update].bind(xo, block_x)
s[s_update].bind(xi, thread_x)

print(tvm.lower(s, [X, s_scan], simple_mode=True))
