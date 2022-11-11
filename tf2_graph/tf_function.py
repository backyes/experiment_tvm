import tensorflow as tf
import timeit

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

# https://zhuanlan.zhihu.com/p/59482934

# mac 
# conda install -c apple tensorflow-deps --force
# python3 -m pip install tensorflow-macos

def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y


def mnist_dataset():
  (x, y), _ = tf.keras.datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds

train_dataset = mnist_dataset()
model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(64)))
model.build()
optimizer = tf.keras.optimizers.Adam()
compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


def training_N_steps(train_ds):
  step = 0
  loss = 0.0
  accuracy = 0.0
  for x, y in train_ds:
    step += 1
    loss = train_one_step(model, optimizer, x, y)
    if tf.equal(step % 10, 0):
      tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
  return step, loss, accuracy


@timing
def train_without_graph(train_dataset, model, optimizer):
  return training_N_steps(train_dataset)

@timing
@tf.function
def train(train_dataset, model, optimizer):
  return training_N_steps(train_dataset)

print("train_with_graph")
step, loss, accuracy = train(train_dataset, model, optimizer)
print("train_without_graph")
step, loss, accuracy = train_without_graph(train_dataset, model, optimizer)
print('Final step', step, ': loss', loss, '; accuracy', compute_accuracy.result())

# ...
# Step 180 : loss 0.266163617 ; accuracy 0.857777774
# Step 190 : loss 0.171451673 ; accuracy 0.861105263
# Step 200 : loss 0.201522663 ; accuracy 0.864500046
# func:'train' args:[(<BatchDataset element_spec=(TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))>, <keras.engine.sequential.Sequential object at 0x1554e24c0>, <keras.optimizers.optimizer_v2.adam.Adam object at 0x1554da7c0>), {}] took: 1.2009 sec
# train_without_graph
# Step 10 : loss 0.294057727 ; accuracy 0.867428541
# Step 20 : loss 0.204026029 ; accuracy 0.871000051
# Step 30 : loss 0.195730701 ; accuracy 0.874000072
# Step 40 : loss 0.0834955424 ; accuracy 0.876375
# Step 50 : loss 0.184545591 ; accuracy 0.878920078
# Step 60 : loss 0.129878417 ; accuracy 0.8815
# Step 70 : loss 0.19208622 ; accuracy 0.883851826
# Step 80 : loss 0.203392774 ; accuracy 0.88614285
# Step 90 : loss 0.177937448 ; accuracy 0.888310373
# Step 100 : loss 0.258376777 ; accuracy 0.890433371
# Step 110 : loss 0.196487188 ; accuracy 0.892193496
# Step 120 : loss 0.17712763 ; accuracy 0.89384377
# Step 130 : loss 0.151500717 ; accuracy 0.894969702
# Step 140 : loss 0.182725728 ; accuracy 0.89685297
# Step 150 : loss 0.214274511 ; accuracy 0.898228586
# Step 160 : loss 0.153123364 ; accuracy 0.899833322
# Step 170 : loss 0.153443336 ; accuracy 0.901297271
# Step 180 : loss 0.122936964 ; accuracy 0.90286839
# Step 190 : loss 0.118593425 ; accuracy 0.903974354
# Step 200 : loss 0.135674149 ; accuracy 0.90535
# func:'train_without_graph' args:[(<BatchDataset element_spec=(TensorSpec(shape=(None, 28, 28), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int64, name=None))>, <keras.engine.sequential.Sequential object at 0x1554e24c0>, <keras.optimizers.optimizer_v2.adam.Adam object at 0x1554da7c0>), {}] took: 2.2060 sec
# Final step 200 : loss tf.Tensor(0.13567415, shape=(), dtype=float32) ; accuracy tf.Tensor(0.90535, shape=(), dtype=float32)
