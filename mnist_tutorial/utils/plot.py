from matplotlib import pyplot as plt

import numpy as np


def visualize_img_data(img_data):
    """Display the first nine images of the provided data

    Args:
        img_data (np.array): A matrix of image data, of shape (N, H, W)
    """
    N, _, _ = img_data.shape
    selected_data = img_data[np.random.choice(N, 9)]

    fig = plt.figure(figsize=(10, 10))
    rows = 3
    columns = 3
    for i in range(1, rows*columns + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(selected_data[i-1])
    
    plt.show()


def plot_histo_chart(hist):
    plt.subplot(121)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.plot(hist.history.get('acc', []), '-o')
    plt.plot(hist.history.get('val_acc', []), '-o')

    plt.subplot(122)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.plot(hist.history.get('loss', []), '-o')
    plt.plot(hist.history.get('val_loss', []), '-o')
    
    plt.show()