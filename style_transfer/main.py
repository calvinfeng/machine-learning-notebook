from pretrained_models.squeeze_net import SqueezeNet
from tensorflow.python.framework.errors_impl import NotFoundError
from image_utils import preprocess_image, load_image
from loss import content_loss

import tensorflow as tf
import matplotlib.pyplot as plt


CKPT_PATH = './pretrained_models/squeezenet.ckpt'


def display_content_and_style(content_img, style_img):
    _, ax_arr = plt.subplots(1, 2)
    ax_arr[0].axis('off')
    ax_arr[1].axis('off')
    ax_arr[0].set_title('Content Source')
    ax_arr[1].set_title('Style Source')
    ax_arr[0].imshow(content_img)
    ax_arr[1].imshow(style_img)
    plt.show()


def main():
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    try:
        model = SqueezeNet(ckpt_path=CKPT_PATH, sess=sess)
    except NotFoundError:
        raise ValueError('checkpoint file is not found, please check %s' % CKPT_PATH)

    content_img = preprocess_image(load_image('images/san_francisco.jpg', size=192))
    style_img = preprocess_image(load_image('images/starry_night.jpg', size=192))
    content_layer = 3

    # Using [None] at the end of a np.array will cast (H, W, C) into (1, H, W, C)
    content_img = content_img[None]
    content_feats = sess.run(model.extract_features()[content_layer], {
        model.image: content_img
    })

    zero_img = tf.zeros(content_img.shape)
    zero_feats = model.extract_features(zero_img)[content_layer]

    content_weight = 6e-2
    loss = sess.run(content_loss(content_weight, content_feats, zero_feats))
    print 'Content loss %f' % loss


if __name__ == '__main__':
    main()