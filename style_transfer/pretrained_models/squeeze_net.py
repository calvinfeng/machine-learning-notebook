import tensorflow as tf


NUM_CLASSES = 1000

class SqueezeNet(object):
    def __init__(self, ckpt_path=None, sess=None):
        """Instantiates a SqueezeNet model

        Args:
            ckpt_path (str): Path to TensorFlow checkpoint files
            sess: TensorFlow session
        """
        self.image = tf.placeholder('float', shape=[None, None, None, 3], name='input_image')
        self.labels = tf.placeholder('int32', shape=[None], name='labels')
        self.layers = self.extract_features(self.image, reuse=False)
        self.features = self.layers[-1]
        
        x = None
        with tf.variable_scope('classifier'):
            with tf.variable_scope('layer0'):
                x = self.features
                self.layers.append(x)
            
            with tf.variable_scope('layer1'):
                W = tf.get_variable('weights', shape=[1, 1, 512, 1000])
                b = tf.get_variable('bias', shape=[1000])
                x = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID')
                x = tf.nn.bias_add(x, b)
                self.layers.append(x)
            
            with tf.variable_scope('layer2'):
                x = tf.nn.relu(x)
                self.layers.append(x)
            
            with tf.variable_scope('layer3'):
                x = tf.nn.avg_pool(x, [1, 13, 13, 1], strides=[1, 13, 13, 1], padding='VALID')
                self.layers.append(x)
            
        self.classifier = tf.reshape(x, [-1, NUM_CLASSES])
        if ckpt_path is not None:
            saver = tf.train.Saver()
            saver.restore(sess, ckpt_path)
        
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, NUM_CLASSES), 
                                                    logits=self.classifier)
        )

    def extract_features(self, input=None, reuse=True):
        """Extracts all feature layers from the SqueezeNet architecture.
        
        Args:
            input (Tensor): A tensor that represents an input image
            resuse (bool): Whether variables within variable scope should be reused
        """
        if input is None:
            input = self.image
        
        layers = []
        x = input  # x is the input to each layer of the network
        with tf.variable_scope('features', reuse=reuse):
            with tf.variable_scope('layer0'):
                W = tf.get_variable('weights', shape=[3, 3, 3, 64])
                b = tf.get_variable('bias', shape=[64])
                x = tf.nn.conv2d(x, W, [1, 2, 2, 1], 'VALID')
                x = tf.nn.bias_add(x, b)
                layers.append(x)
            
            with tf.variable_scope('layer1'):
                x = tf.nn.relu(x)
                layers.append(x)
            
            with tf.variable_scope('layer2'):
                x = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                layers.append(x)
            
            with tf.variable_scope('layer3'):
                x = fire_module(x, 64, 16, 64, 64)
                layers.append(x)
            
            with tf.variable_scope('layer4'):
                x = fire_module(x, 128, 16, 64, 64)
                layers.append(x)

            with tf.variable_scope('layer5'):
                x = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                layers.append(x)            

            with tf.variable_scope('layer6'):
                x = fire_module(x, 128, 32, 128, 128)
                layers.append(x)
            
            with tf.variable_scope('layer7'):
                x = fire_module(x, 256, 32, 128, 128)
                layers.append(x)
            
            with tf.variable_scope('layer8'):
                x = tf.nn.max_pool(x, [1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
                layers.append(x)
            
            with tf.variable_scope('layer9'):
                x = fire_module(x, 256, 48, 192, 192)
                layers.append(x)
            
            with tf.variable_scope('layer10'):
                x = fire_module(x, 384, 48, 192, 192)
                layers.append(x)
            
            with tf.variable_scope('layer11'):
                x = fire_module(x, 384, 64, 256, 256)
                layers.append(x)
            
            with tf.variable_scope('layer12'):
                x = fire_module(x, 512, 64, 256, 256)
                layers.append(x)
            
            return layers


def fire_module(x, x_dim, s11_dim, e11_dim, e33_dim):
    """Create a fire module, which is the the essential building block of a SqueezeNet. A fire
    module is composed of a squeeze  convolution layer which has only 1x1 filter, feeding into an
    expand layer that has a mix of 1x1 and 3x3 convolution filters.

    More information: https://arxiv.org/pdf/1602.07360.pdf

    Args:
        x (tensor): Input tensor
        x_dim (int): Shape[1] of the input tensor
        s11_dim (int): Number of 1x1 filters in the squeeze layer
        e11_dim (int): Number of 1x1 filters in the expand layer
        e33_dim (int): Number of 3x3 filters in the expand layer 
    """
    with tf.variable_scope('fire'):
        with tf.variable_scope('squeeze'):
            W = tf.get_variable('weights', shape=[1, 1, x_dim, s11_dim])
            b = tf.get_variable('bias', shape=[s11_dim])
            s11 = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'VALID') + b
            s11 = tf.nn.relu(s11)

        with tf.variable_scope('e11'):
            W = tf.get_variable('weights', shape=[1, 1, s11_dim, e11_dim])
            b = tf.get_variable('bias', shape=[e11_dim])
            e11 = tf.nn.conv2d(s11, W, [1, 1, 1, 1], 'VALID') + b
            e11 = tf.nn.relu(e11)

        with tf.variable_scope('e33'):
            W = tf.get_variable('weights', shape=[3, 3, s11_dim, e33_dim])
            b = tf.get_variable('bias', shape=[e33_dim])
            e33 = tf.nn.conv2d(s11, W, [1, 1, 1, 1], 'SAME') + b
            e33 = tf.nn.relu(e33)
        
        return tf.concat([e11, e33], 3)
