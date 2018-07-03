import tensorflow as tf


def content_loss(content_weight, current, target):
    """Compute the content loss for style transfer

    The goal is to generate an image that reflects the content of one image and the style of another
    by incorporating both in the loss function. Content loss measures how much the feature map of 
    the generated image differs from the feature map of the source image. 

    Args:
        content_weight (float): Scalar constant to multiply the content loss
        current (Tensor): Features of the current image, with shape (1, H, W, C)
        target (Tensor): Features of the content image, with shape (1, H, W, C)
    
    Returns:
        content_loss (float)
    """
    H = tf.shape(current)[1]
    W = tf.shape(current)[2]
    C = tf.shape(current)[3]

    current = tf.reshape(current, (H*W, C))
    target = tf.reshape(target, (H*W, C))
    loss = tf.reduce_sum((current - target) ** 2)

    return content_weight * loss


def gram_matrix(features, normalize=True):
    """Compute Gram matrix from feature tensor

    Args:
        features (Tensor): Tensor of shape (1, H, W, C) giving features for a single image
        normalize (bool): Whether to normalize the Gram matrix. If true, divide the matrix by the number of neurons
    """
    H = tf.shape(features)[1]
    W = tf.shape(features)[2]
    C = tf.shape(features)[3]

    # Computing Gram matrix is very similar to computing covariance matrix
    features = tf.reshape(features, (H*W, C))
    features_t = tf.transpose(features)
    
    gram = tf.matmul(features_t, features)
    if normalize:
        gram = tf.divide(gram, tf.cast(H*W*C, tf.float32))
    

    return gram


def style_loss():
    pass
