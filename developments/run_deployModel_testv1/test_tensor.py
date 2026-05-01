# check_tf.py
import tensorflow as tf

print("TF version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]