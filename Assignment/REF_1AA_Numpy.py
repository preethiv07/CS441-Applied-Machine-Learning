%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
img_arr = plt.imread('04_parrot.jpg')

# The image dimensions
# img_arr.shape

# The red channel
red_channel = img_arr[:, :, 0]
assert red_channel.shape == (700, 900)

# The green channel
green_channel = img_arr[:, :, 1]
assert green_channel.shape == (700, 900)

# The blue channel
blue_channel = img_arr[:, :, 2]
assert blue_channel.shape == (700, 900)


fig, axes = plt.subplots(1, 3, dpi=144, figsize=(3.6*3, 2.8), sharex=True, sharey=True)

ax = axes[0]
ax.imshow(red_channel, cmap='gray')
ax.set_title('The Red Channel')

ax = axes[1]
ax.imshow(green_channel, cmap='gray')
ax.set_title('The Green Channel')

ax = axes[2]
ax.imshow(blue_channel, cmap='gray')
ax.set_title('The Blue Channel');

#Luminance
# 𝑌=0.2126𝑅+0.7152𝐺+0.0722𝐵

# Challenge: Take this colored image, and turn it into a gray-scaled image. Do this in four different way.

# Approach 1
gray1 = 0.2126 * red_channel + 0.7152 * green_channel + 0.0722 * blue_channel
plt.imshow(gray1, cmap='gray');