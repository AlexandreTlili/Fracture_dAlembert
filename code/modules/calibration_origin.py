import matplotlib.pyplot as plt
import numpy as np

import PIL.Image as Image

def find_origin_and_scale(image):

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='hot', origin="lower")

    # Get position of zero and two points
    fig.suptitle("Select x-origin, then two points of known distance.")
    newOrigin, point1, point2 = plt.ginput(n=3, timeout=1000)
    distPix = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    plt.close(fig)

    # Get the user input for the distance
    distUser = input("Distance between the two points [in cm]: ")
    distUser = float(distUser) / 100

    scaleRatio_pix_to_real = distUser / distPix
    print(f'Pix to real ratio: {scaleRatio_pix_to_real}')
    print(f"New origin {newOrigin}")

    return newOrigin[0], scaleRatio_pix_to_real

