from img_data_utils import *
from matplotlib import pyplot as plt


def main():
    dict = load_jpg_from_dir("datasets/dog-vs-cat-train/", num_images_per_class=5000)
    print dict['X'].shape
    print dict['y'].shape
    # plt.subplot(211)
    # plt.imshow(img)
    # plt.show()


if __name__ == '__main__':
    main()
