import tensorflow as tf

@tf.function(jit_compile=True)
def running_exapmle(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

x = tf.random.uniform((16834, 16834))
y = tf.random.uniform((16834, 16834))

print(running_exapmle(x, y))
