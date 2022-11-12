'''
OUTPUTDIR=./outputs_tf_fmnst/wx_`date +%Y_%m_%d_%H_%m_%S`; mkdir -p $OUTPUTDIR ; TF_XLA_FLAGS=--tf_xla_clustering_debug  TF_CPP_MAX_VLOG_LEVEL=5 TF_DUMP_GRAPH_PREFIX=$OUTPUTDIR/graphs XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit --xla_dump_to=$OUTPUTDIR/hlo --xla_dump_hlo_as_html --xla_dump_fusion_visualization" LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$VCONDA_PREFIX/lib  python    tf_fmnst.py --epochs 1  --xla 2>&1 | tee $OUTPUTDIR/output.log
'''
import tensorflow as tf
import argparse
import timeit
import time
import pandas as pd
parser = argparse.ArgumentParser(description='To show xla flow.')
parser.add_argument("--epochs", type=int, default=10, help="maximum number of epochs to run. ")
parser.add_argument("--iters", type=int, default=4, help="maximum number of iterations per epoch to run. ")
parser.add_argument("--xla", action='store_true', default=False, help="enable xla")
parser.add_argument("--warmup", action='store_true', default=False, help="run warmup")
parser.add_argument('-f', type=str, help='log file')
parser.add_argument('--logdir', type=str, help='log directory')
parser.add_argument('--bs', type=int, default=256, help='batch size')
parser.add_argument("--profile", action='store_true', default=False, help="run profiler")
args = parser.parse_args()
# Check that GPU is available: cf. https://colab.research.google.com/notebooks/gpu.ipynb
assert(tf.test.gpu_device_name())
tf.keras.backend.clear_session()
if args.xla:
 #tf.config.optimizer.set_jit(True) 
 tf.config.optimizer.set_jit("autoclustering") 
else:
 tf.config.optimizer.set_jit(False)
def load_data():
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_train = x_train.astype('float32') / args.bs
  x_test = x_test.astype('float32') / args.bs
# Convert class vectors to binary class matrices.
  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
  return ((x_train, y_train), (x_test, y_test))
def generate_model():
  return tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
#tf.keras.layers.Flatten(input_shape=x_train.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
  ])
class DebugCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin (self, batch, logs=None):
        print (f"======= batch {batch} begin =========") 
    def on_train_batch_end (self, batch, logs=None):
        print (f"======= batch {batch} end =========") 
        
def compile_model(model):
  opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
  return model
def train_model(model, x_train, y_train, x_test, y_test, epochs=25):
  model.fit(x_train, y_train, batch_size=args.bs, epochs=epochs, verbose=2, 
            validation_steps=0, validation_data=(x_test, y_test), shuffle=True,
            callbacks=[DebugCallback()])
def warmup(model, x_train, y_train, x_test, y_test):
  # Warm up the JIT, we do not wish to measure the compilation time.
  initial_weights = model.get_weights()
  train_model(model, x_train, y_train, x_test, y_test, epochs=1)
  model.set_weights(initial_weights)
(x_train, y_train), (x_test, y_test) = load_data()
model = generate_model()
model = compile_model(model)
model.summary()
if args.warmup:
        warmup(model, x_train, y_train, x_test, y_test)
data_size = args.bs*args.iters
time0 = time.time()
if args.profile:
 tf.profiler.experimental.start(args.logdir+"/profile")
train_model(model, x_train[:data_size], y_train[:data_size], x_test, y_test, epochs=args.epochs)
if args.profile:
 tf.profiler.experimental.stop()
print ('Train time (s):', time.time() - time0)
if False: #  args.xla:
 print ( "hlo ir:")
 print(generate_model.experimental_get_compiler_ir(x,y,z,zz)(stage='hlo'))
 print ( "optimized hlo ir:")
 print(generate_model.experimental_get_compiler_ir(x,y,z,zz)(stage='optimized_hlo'))
 print ( "optimized hlo dot:")
 print(generate_model.experimental_get_compiler_ir(x,y,z,zz)(stage='optimized_hlo_dot'))
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0], 'Test accuracy:', scores[1])
'''
# We need to clear the session to enable JIT in the middle of the program.
tf.keras.backend.clear_session()
tf.config.optimizer.set_jit(True) # Enable XLA.
model = compile_model(generate_model())
(x_train, y_train), (x_test, y_test) = load_data()
warmup(model, x_train, y_train, x_test, y_test)
time0 = time.time()
train_model(model, x_train, y_train, x_test, y_test, epochs=EPOCHS)
print ('Train time (s):', time.time() - time0)
'''
