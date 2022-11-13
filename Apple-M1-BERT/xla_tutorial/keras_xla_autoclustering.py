import timeit

from functools import wraps
from time import time

import tensorflow as tf
import tensorflow_datasets as tfds

# https://www.tensorflow.org/xla/tutorials/autoclustering_xla
# no high performance gains with xla

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        #print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #  (f.__name__, args, kw, te-ts))
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap


# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert(tf.test.gpu_device_name())

tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(False) # Start with XLA disabled.

def load_data():
  result = tfds.load('cifar10', batch_size = -1)
  (x_train, y_train) = result['train']['image'],result['train']['label']
  (x_test, y_test) = result['test']['image'],result['test']['label']

  x_train = x_train.numpy().astype('float32') / 256
  x_test = x_test.numpy().astype('float32') / 256

  # Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()


# def generate_model():
#   return tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Conv2D(32, (3, 3)),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
# 
#     tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Conv2D(64, (3, 3)),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Dropout(0.25),
# 
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512),
#     tf.keras.layers.Activation('relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Activation('softmax')
#   ])

def generate_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(8, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(8, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(8, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(8, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
  ])

model = generate_model()


#@tf.function
def compile_model(model):
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model

model = compile_model(model)

@timing
#@tf.function can not be added here
def train_model(model, x_train, y_train, x_test, y_test, epochs=1):
  model.fit(x_train, y_train, batch_size=256, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)

print("warmup")
warmup(model, x_train, y_train, x_test, y_test)
print("train")
train_model(model, x_train, y_train, x_test, y_test)

print("evaluate")
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


print("=============jit===========")

# We need to clear the session to enable JIT in the middle of the program.
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
model = compile_model(generate_model())
(x_train, y_train), (x_test, y_test) = load_data()

print("warmup")
warmup(model, x_train, y_train, x_test, y_test)
print("train")
train_model(model, x_train, y_train, x_test, y_test)

print("evaluate")
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
