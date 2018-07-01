from pretrained_models.squeeze_net import SqueezeNet
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError


def get_session():
    """Create a session that dynamically allocates memory
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config=config)

    return session


CKPT_PATH = './pretrained_models/squeezenet.ckpt'


def main():
    tf.reset_default_graph()
    sess = get_session()
    
    try:
        model = SqueezeNet(ckpt_path=CKPT_PATH, sess=sess)
    except NotFoundError:
        raise ValueError('checkpoint file is not found, please check %s' % CKPT_PATH)


if __name__ == '__main__':
    main()