import timeit
import click
import tensorflow as tf


@click.command()
@click.option('--device', required=True, help="device will be used, [cpu] or [gpu]")
def main(device):
    if device == "cpu":
        tf.config.experimental.set_visible_devices([], 'GPU')
    elif device == "gpu":
        tf.device('/gpu:0')
    else: raise Exception("Unknown devices")    


    model = tf.keras.Sequential((
        tf.keras.layers.Reshape(target_shape=(200 * 200,), input_shape=(200, 200)),
        # if set 4096 => 40960, mac gpu is exhausted
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(8192, activation='relu'),
        tf.keras.layers.Dense(64)))
    model.build()

    
    @tf.function
    def graph_model(input_data):
      return model(input_data)


    input_data = tf.zeros([1, 200, 200])
    # Warm up
    model(input_data); graph_model(input_data)
    print("Eager conv:", timeit.timeit(lambda: model(input_data), number=10))
    print("Function conv:", timeit.timeit(lambda: graph_model(input_data), number=10))
    print("Note how there's not much difference in performance for convolutions")

if __name__ == "__main__":
    main()

# (tvm-m1) shwangyanfei@wangyanfeideMBP tf2_graph % python3 autograph2.py --device=gpu
# Metal device set to: Apple M1
# 
# systemMemory: 16.00 GB
# maxCacheSize: 5.33 GB
# 
# 2022-11-11 18:51:42.644585: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:306] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
# 2022-11-11 18:51:42.644695: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:272] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
# 2022-11-11 18:51:45.296288: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
# 2022-11-11 18:51:45.296560: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type GPU is enabled.
# Eager conv: 0.04636283299999988
# Function conv: 0.027340999999999838
# Note how there's not much difference in performance for convolutions
# (tvm-m1) shwangyanfei@wangyanfeideMBP tf2_graph %
