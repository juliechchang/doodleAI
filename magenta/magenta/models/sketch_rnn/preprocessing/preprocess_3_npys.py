# -*- coding: utf-8 -*-

import cv2
import sys
import numpy as np

def scale_image(infile, size=(48, 48), outfile=None):
    # read image
    img = cv2.imread(infile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

    h, w = img.shape[:2]
    sh, sw = size

    # aspect ratio of image
    aspect = w/h

    # padding
    pad = [0, 0, 0, 0] # (top, left, bottom, right)

    new_h, new_w = sh, sw

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad[0] = np.floor(pad_vert).astype(int)
        pad[2] = np.ceil(pad_vert).astype(int)

    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad[1] = np.floor(pad_horz).astype(int)
        pad[3] = np.ceil(pad_horz).astype(int)

    # scale and pad
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], borderType=cv2.BORDER_CONSTANT, value=0)

    # increase contrast
    img[img > 0] = 255

    # display or save as npy
    if not outfile:
        cv2.imshow('scaled image', img)
        cv2.waitKey(0)
    else:
        np.save(outfile, img)

if __name__ == '__main__':
    scale_image(infile=sys.argv[1], outfile='sample.npy')
