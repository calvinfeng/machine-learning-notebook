from pretrained_models.squeeze_net import SqueezeNet
from tensorflow.python.framework.errors_impl import NotFoundError
from image_utils import preprocess_image, load_image, deprocess_image
from loss import content_loss, gram_matrix, style_loss, total_variation_loss

import tensorflow as tf
import matplotlib.pyplot as plt


CKPT_PATH = './pretrained_models/squeezenet.ckpt'


def display_content_and_style(content_img, style_img):
    _, ax_arr = plt.subplots(1, 2)
    ax_arr[0].axis('off')
    ax_arr[1].axis('off')
    ax_arr[0].set_title('Content Source')
    ax_arr[1].set_title('Style Source')
    ax_arr[0].imshow(deprocess_image(content_img))
    ax_arr[1].imshow(deprocess_image(style_img))
    plt.show()


def style_transfer(content_img_path, img_size, style_img_path, style_size, content_layer, content_weight, style_layers,
    style_weights, tv_weight, init_random=False):
    """Perform style transfer from style image to source content image
    
    Args:
        content_img_path (str): File location of the content image.
        img_size (int): Size of the smallest content image dimension.
        style_img_path (str): File location of the style image.
        style_size (int): Size of the smallest style image dimension.
        content_layer (int): Index of the layer to use for content loss.
        content_weight (float): Scalar weight for content loss.
        style_layers ([]int): Indices of layers to use for style loss.
        style_weights ([]float): List of scalar weights to use for each layer in style_layers.
        tv_weigh (float): Scalar weight of total variation regularization term.
        init_random (boolean): Whether to initialize the starting image to uniform random noise.
    """
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    try:
        model = SqueezeNet(ckpt_path=CKPT_PATH, sess=sess)
    except NotFoundError:
        raise ValueError('checkpoint file is not found, please check %s' % CKPT_PATH)

    # Extract features from content image
    content_img = preprocess_image(load_image(content_img_path, size=img_size))
    content_feats = model.extract_features(model.image)
    
    # Create content target
    content_target = sess.run(content_feats[content_layer], {model.image: content_img[None]})

    # Extract features from style image
    style_img = preprocess_image(load_image(style_img_path, size=style_size))
    style_feats_by_layer = [content_feats[i] for i in style_layers]

    # Create style targets
    style_targets = []
    for style_feats in style_feats_by_layer:
        style_targets.append(gram_matrix(style_feats)) 
    style_targets = sess.run(style_targets, {model.image: style_img[None]})

    if init_random:
        generated_img = tf.Variable(tf.random_uniform(content_img[None].shape, 0, 1), name="image")
    else:
        generated_img = tf.Variable(content_img[None], name="image")

    # Extract features from generated image
    current_feats = model.extract_features(generated_img)

    loss = content_loss(content_weight, current_feats[content_layer], content_target) + \
        style_loss(current_feats, style_layers, style_targets, style_weights) + \
        total_variation_loss(generated_img, tv_weight)

    # Set up optimization parameters
    init_learning_rate = 3.0
    decayed_learning_rate = 0.1
    max_iter = 200

    learning_rate = tf.Variable(init_learning_rate, name="lr")
    with tf.variable_scope("optimizer") as opt_scope:
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[generated_img])
    
    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=opt_scope.name)
    sess.run(tf.variables_initializer([learning_rate, generated_img] + opt_vars))
    
    # Create an op that will clamp the image values when run
    clamp_image_op = tf.assign(generated_img, tf.clip_by_value(generated_img, -1.5, 1.5))
    
    display_content_and_style(content_img, style_img)

    for t in range(max_iter):
        sess.run(train_op)
        if t < int(0.90 * max_iter):
            sess.run(clamp_image_op)
        elif t == int(0.90 * max_iter):
            sess.run(tf.assign(learning_rate, decayed_learning_rate))
        
        if t % 20 == 0:
            current_loss = sess.run(loss)
            print 'Iteration %d: %f' % (t, current_loss) 
        
    img = sess.run(generated_img)
    plt.imshow(deprocess_image(img[0], rescale=True))
    plt.axis('off')
    plt.show()


def main():
    params = {
        'content_img_path': 'images/san_francisco.jpg',
        'img_size': 192,
        'style_img_path': 'images/starry_night.jpg',
        'style_size': 192,
        'content_layer': 3,
        'content_weight': 5e-2,
        'style_layers': [1, 4, 6, 7],
        'style_weights': [200000, 500, 12, 1],
        'tv_weight': 5e-2
    }

    style_transfer(**params)


if __name__ == '__main__':
    main()