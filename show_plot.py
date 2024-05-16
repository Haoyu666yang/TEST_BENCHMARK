#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def display_images(directory):

    for filename in os.listdir(directory):

        if filename.endswith('.png'):
            filepath = os.path.join(directory, filename)
            print(f"Displaying image: {filepath}")

            img = mpimg.imread(filepath)
            plt.imshow(img)
            plt.title(filename)
            plt.axis('off') 
            plt.show()

display_images('.')
