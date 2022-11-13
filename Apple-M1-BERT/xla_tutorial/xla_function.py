
import numpy as np
import click
import os
import tensorflow as tf
import time
#tf.compat.v1.enable_eager_execution()

#https://www.tensorflow.org/xla/tutorials/jit_compile



@click.command()
@click.option('--device', required=True, help="device will be used, [cpu] or [gpu]")
def main(device):
    if device == "cpu":
        tf.config.experimental.set_visible_devices([], 'GPU')
    elif device == "gpu": tf.device('/gpu:0')
    else:
        raise Exception("Unknown devices")

    # Size of each input image, 28 x 28 pixels
    IMAGE_SIZE = 28 * 28
    # Number of distinct number labels, [0..9]
    NUM_CLASSES = 10
    # Number of examples in each training batch (step)
    TRAIN_BATCH_SIZE = 100
    # Number of training steps to run
    TRAIN_STEPS = 1000
        
    
    
    # Casting from raw data to the required datatypes.
    def cast(images, labels):
      images = tf.cast(
          tf.reshape(images, [-1, IMAGE_SIZE]), tf.float32)
      labels = tf.cast(labels, tf.int64)
      return (images, labels)
    
    # Loads MNIST dataset.
    train, test = tf.keras.datasets.mnist.load_data()
    train_ds = tf.data.Dataset.from_tensor_slices(train).batch(TRAIN_BATCH_SIZE).repeat()
    
    layer = tf.keras.layers.Dense(NUM_CLASSES)
    optimizer = tf.keras.optimizers.Adam()

    #@tf.function
    @tf.function(jit_compile=True)
    def train_mnist(images, labels):
        images, labels = cast(images, labels)
    
        with tf.GradientTape() as tape:
          predicted_labels = layer(images)
          loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
              logits=predicted_labels, labels=labels
          ))
        layer_variables = layer.trainable_variables
        grads = tape.gradient(loss, layer_variables)
        optimizer.apply_gradients(zip(grads, layer_variables))

    for images, labels in train_ds:
      if optimizer.iterations > TRAIN_STEPS:
        break
      train_mnist(images, labels)
    
    images, labels = cast(test[0], test[1])
    predicted_labels = layer(images)
    correct_prediction = tf.equal(tf.argmax(predicted_labels, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Prediction accuracy after training: %s" % accuracy)
    
    print(train_mnist.experimental_get_compiler_ir(images, labels)(stage='hlo'))


if __name__ == "__main__":
   main()

