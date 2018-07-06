import tensorflow as tf

# The goal is to generate an image that reflects the content of one image and the style of another
# by incorporating both in the loss function. 
def content_loss(content_weight, current_features, target_features):
    """Compute the content loss for style transfer

    Content loss measures how much the feature map of the generated image differs from the feature 
    map of the source image. 

    Args:
        content_weight (float): Scalar constant to multiply the content loss
        current_features (Tensor): Features of the current_features image, with shape (1, H, W, C)
        target_features (Tensor): Features of the content image, with shape (1, H, W, C)
    
    Returns:
        content_loss (float)
    """
    H = tf.shape(current_features)[1]
    W = tf.shape(current_features)[2]
    C = tf.shape(current_features)[3]

    current_features = tf.reshape(current_features, (H*W, C))
    target_features = tf.reshape(target_features, (H*W, C))
    loss = tf.reduce_sum((current_features - target_features) ** 2)

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


def style_loss(features, style_layers, style_targets, style_weights):
    """Compute style loss at a set of layers

    Args:
        features (Tensor): List of features at every layer of the current image, as produced by the extract_features
                           function.
        style_layers ([]int): List of layer indices into features giving the layers to include in the loss function.
        style_targets ([]Tensor): List of the same length as style_layers, where style_targets[i] is a Tensor giving the
                                  Gram matrix the source style image computed at layer style_layers[i].
        style_weights ([]float): List of the same length as style_layers, where style_weights[i] is a scalar giving the
                                 the weight for the style loss at layer style_layers[i].

    Returns:
        style_loss (Tensor): A tensor containing the scalar style loss.
    """
    style_loss = 0
    
    for i in range(len(style_layers)):
        gram = gram_matrix(features[style_layers[i]])
        gram_target = style_targets[i]
        loss = (gram - gram_target)**2
        style_loss += tf.reduce_sum(loss) * style_weights[i]
    
    return style_loss


def total_variation_loss(img, tv_weight):
    """Compute total variation loss

    Adding an extra regularization term to the loss function will encourage smoothness in the final output image. 

    Args:
        img (Tensor): Tensor of shape (1, H, W, 3), holding an input image.
        tv_weight (float): Scalar giving the weight to use for the total variation loss.

    Returns:
        loss (Tensor) Tensor holding a scalar giving the total variation loss for img weighted by tv_weight.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    img = tf.reshape(img, (H, W, 3))
    img_origin_vert_slice = tf.slice(img, [0, 0, 0], [H-1, W, 3])
    img_vert_slice = tf.slice(img, [1, 0, 0], [H-1, W, 3])

    img_origin_horz_slice = tf.slice(img, [0, 0, 0], [H, W-1, 3])
    img_horz_slice = tf.slice(img, [0, 1, 0], [H, W-1, 3])

    loss = tf.reduce_sum((img_origin_vert_slice - img_vert_slice)**2) + \
        tf.reduce_sum((img_origin_horz_slice - img_horz_slice)**2)
    
    return tv_weight * loss
