import numpy as np
import click
from functools import wraps
from time import time

import tensorflow as tf
from dataset import get_dataset, prepare_dataset
from model import get_model

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




@click.command()
@click.option('--device', required=True, help="device will be used, [cpu] or [gpu]")
def main(device):
    if device == "cpu":
        tf.config.experimental.set_visible_devices([], 'GPU')
    elif device == "gpu": tf.device('/gpu:0')
    else:
        raise Exception("Unknown devices")

    dataset = get_dataset("fr-en")
    
    print("Dataset loaded. Length:", len(dataset), "lines")
    
    train_dataset = dataset[0:100000]
    
    print("Train data loaded. Length:", len(train_dataset), "lines")
    
    (encoder_input,
    decoder_input,
    decoder_output,
    encoder_vocab,
    decoder_vocab,
    encoder_inverted_vocab,
    decoder_inverted_vocab) = prepare_dataset(
      train_dataset,
      shuffle = False,
      lowercase = True,
      max_window_size = 20
    )
    
    transformer_model = get_model(
      EMBEDDING_SIZE = 64,
      ENCODER_VOCAB_SIZE = len(encoder_vocab),
      DECODER_VOCAB_SIZE = len(decoder_vocab),
      ENCODER_LAYERS = 2,
      DECODER_LAYERS = 2,
      NUMBER_HEADS = 4,
      DENSE_LAYER_SIZE = 128
    )
    
    
    transformer_model.compile(
      optimizer = "adam",
      loss = [
        "sparse_categorical_crossentropy"
      ],
      metrics = [
        "accuracy"
      ],
      jit_compile=True
    )
    
    transformer_model.summary()
    
    x = [np.array(encoder_input), np.array(decoder_input)]
    y = np.array(decoder_output)
    
    name = "transformer"
    checkpoint_filepath = "./logs/transformer_ep-{epoch:02d}_loss-{loss:.2f}_acc-{accuracy:.2f}.ckpt"
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir = "logs/{}".format(name)
    )
    
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath = checkpoint_filepath,
      monitor = "val_accuracy",
      mode = "max",
      save_weights_only = True,
      save_best_only = True,
      verbose = True
    )
    
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
      monitor = "val_accuracy",
      mode = "max",
      patience = 2,
      min_delta = 0.001,
      verbose = True
    )

    @timing
    def train():
        transformer_model.fit(
          x,
          y,
          epochs = 1,
          batch_size = 32,
          validation_split = 0.1,
          callbacks=[
            model_checkpoint_callback,
            tensorboard_callback,
            early_stopping_callback
          ]
        )


    train()


if __name__ == "__main__":
   main()

