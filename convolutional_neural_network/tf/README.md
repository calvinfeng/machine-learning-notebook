# Tensorflow
## GPU Installations
### CUDA Toolkit 8.0
### Nvidia drivers
### cuDNN v6.0
### `libcupti-dev` library

## `tf.layers` vs `tf.nn`
I noticed that there are `tf.layers.conv2d` and `tf.nn.conv2d`, and they seem to work the same except that they have slightly different API.

This is `tf.nn.conv2d`
```python
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
W = tf.get_variable('W', shape=[7, 7, 3, 32]) # (filter height, filter width, input channels and output channels)
b = tf.get_variable('b1', shape=[32]) # (output channels)

conv_out = tf.nn.conv2d(inputs=X, filters=W, strides=[1, 1, 1, 1], padding='VALID') + b
```

This is `tf.layers.conv2d`
```python
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
conv_out = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[7, 7], padding='SAME', activation=tf.nn.relu,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                           activity_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

```
### VALID vs SAME
`'VALID'` drops the right-most columns or bottom-most rows if the convolution goes out of bound.
```
inputs:         1  2  3  4  5  6  7  8  9  10 11 (12 13)
               |________________|                dropped
                              |_________________|
```

`'SAME'` tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right or extra row to the bottom.
```

               pad|                                      |pad
   inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
               |________________|
                              |_________________|
                                             |________________|
```

By default, all convolution is performed on `NHWC` data format, i.e. N by height by width by channels.

## convolution
The formula for determing output size is `H' = (H - F + 2P) / S` where H is height, F is filter height, and S is stride.
Same applies for computing width.
