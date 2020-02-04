import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import random
from tkinter import *
from scipy import ndimage
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", default=1, type=int,
        help="number of samples to be generated")
args = vars(ap.parse_args())

def create_circle(image, xc, yc, radius, thickness):
    w, h = image.shape
    xx, yy = np.mgrid[:w, :h]
    circle = (xx - xc) ** 2 + (yy - yc) ** 2
    thickness = thickness + 5 * radius
    donut = np.logical_and(circle < radius ** 2 + thickness + 4 * radius,
                           circle > radius ** 2 - thickness + 4 * radius)
    return donut

def generate_raw_data(windows_size):
    w, h = windows_size
    image = np.zeros([w, h])

    circle_nb = random.randint(5, 15)
    xc, yc = w // 2, h // 2

    for k in range(args["number"]):
        print("Working on image #{}".format(k+1))
        r = []
        for i in range(circle_nb):
            radius = random.randint(int(w * 0.05), int(w * 1.1 / 2 ))
            circle = create_circle(image, xc, yc, radius, 30)

            image += circle
            r.append(radius)
        image = ndimage.distance_transform_bf(image)
        plt.matshow(image)
        print("Radius list : ", r)

        # image_noise = image + 0.2 * np.random.randn(*image.shape)
        # plt.matshow(image_noise)
        #
        # image_med = ndimage.median_filter(image_noise, 3)
        # plt.matshow(image_med)

def generate_integrated_data(sector_number, ttheta_value_number, sample_nb=1, save_path=None):
    thickness = 5

    for i in range(sample_nb):
        print("Generating fake image #{}/{}".format(i + 1, sample_nb))
        peak_number = random.randint(5, 15)
        peak_positions = [random.randint(ttheta_value_number // 10, ttheta_value_number - 1) for i in range(peak_number)]
        image = np.zeros([0, ttheta_value_number])
        # image = np.zeros([ttheta_value_number])
        for k in range(sector_number):
            sector = np.zeros([1, ttheta_value_number])
            for position in peak_positions:
                pixel_intensity = random.randint(10, 100)
                thickness = 1 + int(pixel_intensity//40)
                # sector[0, position] = pixel_intensity
                sector[0, position : position + thickness] = [pixel_intensity * (1 - i/thickness) for i in range(thickness)]
                sector[0, position - thickness : position] = [pixel_intensity * (1 - i/thickness) for i in range(thickness)]
                # sector[0, position - thickness -1 : position + thickness +1] = ndimage.distance_transform_edt(sector[0, position - thickness -1 : position + thickness +1])
            sector = ndimage.distance_transform_bf(sector)
            image = np.concatenate((image, sector), axis=0)
        # image = ndimage.distance_transform_bf(image)

        if save_path is not None:
            np.save(save_path + "fake_{}".format(i+1), image)
        else:
            plt.matshow(image)
            plt.show()



if __name__ == "__main__":
    # generate_raw_data(windows_size=(400, 400))
    save_path = "../data/generated_images/"
    generate_integrated_data(1080, 1000, sample_nb=args["number"], save_path=save_path)
    # generate_integrated_data(1080, 1000, sample_nb=args["number"])

    # plt.show()



# im = np.zeros((20, 20))
#
# im[5:-5, 5:-5] = 1
#
# im = ndimage.distance_transform_bf(im)
#
# im_noise = im + 0.2 * np.random.randn(*im.shape)
#
# im_med = ndimage.median_filter(im_noise, 3)
# plt.matshow(im)
# plt.matshow(im_noise)
# plt.matshow(im_med)
# plt.show()

# xx, yy = np.mgrid[:2048, :2048]
# circle = (xx - 1024) ** 2 + (yy - 1024) ** 2
# donut = np.logical_and(circle < (100000 + 100), circle > (100000 - 100))
#
# plt.matshow(donut)
# plt.show()
