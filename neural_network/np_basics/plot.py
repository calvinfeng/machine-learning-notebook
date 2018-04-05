import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

"""Plotting

The most important function in matplotlib is plot which allows you to plot 2D data
"""
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()

# With just a little bit of extra work, we can easily plot multiple lines as once and add a title, legend and axis labels
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()

"""Subplots

You can plot different things in the same figure using the subplot function
"""
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1, and set first such subplot as active
plt.subplot(2, 1, 1)
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active and make the second plot
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()

img = imread('assets/cat.jpg')
img_tinted = img * [1, 0.95, 0.9]

# Activate first subplot
plt.subplot(1, 2, 1)
plt.imshow(img)
# Activate second subplot
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img_tinted))

# A slight gotcha with imshow is that it might give starnge results if presented with data that is not uint8. To work around
# this, we explicitly cast the image to uint8 before displaying it
plt.show()
