import numpy as np
import snoopy

test():
    np.random.seed(0)
    n = 100
    a = np.random.normal(size=n).astype(np.float32)
    b = np.random.normal(size=n).astype(np.float32)
    c = a + b
