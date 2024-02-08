import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
import time

def read_file(pathImage):
    assert os.path.isfile(pathImage), f"Non-existing image at {pathImage}"
    return Image.open(pathImage)

def crop_from_coords(imagePIL, xMin=None, xMax=None, yMin=None, yMax=None, show=False):
    """ Crop image with origin upper-left """
    if xMin is None:
        xMin = 0
    if xMax is None:
        xMax = imagePIL.size[0]
    if yMin is None:
        yMin = 0
    if yMax is None:
        yMax = imagePIL.size[1]

    #left, upper, right, and lower (with origin upper-left)
    cropped = imagePIL.crop((xMin, yMin, xMax, yMax))

    if show:
        cropped.show()

    return cropped


def cropBox_from_GUI(imagePIL):
    fig, ax = plt.subplots()

    ax.imshow(np.asarray(imagePIL), origin='upper')

    fig.suptitle('Point the box extend (two point)')
    point1, point2 = plt.ginput(n=2, timeout=30)

    plt.close()

    xMin, xMax = point1[0], point2[0]
    yMin, yMax = point1[1], point2[1]

    xMin, xMax = sorted([xMin, xMax])
    yMin, yMax = sorted([yMin, yMax])

    return {"xMin": xMin, "yMin": yMin, 
            "xMax": xMax, "yMax": yMax}

def convert_to_grey(imagePIL):
    return imagePIL.convert('L')

def offset_scale_from_std(imagePIL, n_std_min=2, n_std_max=3):
    image_arr = np.asarray(imagePIL)
    offset_contrast = image_arr.mean() - n_std_min * image_arr.std()
    scale_contrast = 256 / (n_std_max + n_std_min) / image_arr.std()
    return offset_contrast, scale_contrast


def change_intensity(imagePIL, offset=0, scaleFactor=1., show=False):
    func = (lambda greylevel: scaleFactor * (greylevel - offset))
    contrasted = imagePIL.point(func)

    if show:
        contrasted.show()
    return contrasted

def gaussian_blur(imagePIL, radiusKernel=2):
    return imagePIL.filter(ImageFilter.GaussianBlur(radius=radiusKernel))


def test():
    image = read_file("/media/anais/Data/Alexandre/data/testPolymer/test2/snapshots/Basler_acA2440-75um__23178718__20240207_173847780_1.tiff")
    image.show()
    time.sleep(0.5)
    print(image.size)

    crop_bounds = cropBox_from_GUI(image)
    cropped = crop_from_coords(image, **crop_bounds)
    cropped.show()
    time.sleep(0.5)

    #blurred = gaussian_blur(cropped, radiusKernel=2)
    blurred = cropped
    blurred.show()
    time.sleep(1)

    param_contrast = offset_scale_from_std(blurred, n_std_min=2, n_std_max=3)

    contrasted = change_intensity(blurred, *param_contrast)
    contrasted.show()

    contrasted_array = np.asarray(contrasted)
    contrasted_min, contrasted_max = contrasted_array.min(), contrasted_array.max()
    print(f"Contrasted: {contrasted_min, contrasted_max}")
