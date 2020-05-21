# -*- coding: utf-8 -*-
# @Author  : ZillyRex


import os
from multiprocessing import Pool, cpu_count
import cv2.cv2 as cv2


def resize_img(resize_func, path_img, path_img_out,
               path_anno=None, path_anno_out=None,
               path_label=None, path_label_out=None, verbose=0):
    """
    Resize an image and corresponding annotation and/or label file.

    Args:
        resize_func: A function defining the resize strategy which calls two
        parameters, width and height, and returns the new width and height after resizing.
        eg. resize_func(x, y): return (x/2 if x > 2000 else x, y/3 if y > 3000 else y)
        path_img: Path of the image you wanna resize.
        path_img_out: Path of the folder you wanna save the image after resizing.
        path_anno: Path of the annotation file corresponding to the image.
        path_anno_out: Path of the folder you wanna save the annotation file after resizing.
        path_label: Path of the label file corresponding to the image.
        path_label_out: Path of the folder you wanna save the label file after resizing.

    Returns:
        None
    """
    img = cv2.imread(path_img)
    if not img:
        print('Something wrong with the image {}.'.format(path_img))
        return
    H, W, _ = img.shape
    img = cv2.resize(img, (resize_func(W, H)), interpolation=cv2.INTER_NEAREST)

    if not os.path.isdir(path_img_out):
        os.mkdir(path_img_out)
    basename = os.path.basename(path_img)
    cv2.imwrite(os.path.join(path_img_out, basename), img)

    if path_anno:
        if path_anno_out:
            pass
            # TODO: change anno and save it.
        else:
            print('You need to assign the path_anno_out.')
    else:
        print('No path_anno assigned.')

    if path_label:
        if path_label_out:
            pass
            # TODO: change label and save it.
        else:
            print('You need to assign the path_label_out.')
    else:
        print('No path_label assigned.')

    if verbose:
        print(basename)


def resize_imgs(resize_func, path_img_folder, path_img_out,
                path_anno_folder=None, path_anno_out=None,
                path_label_folder=None, path_label_out=None, verbose=0):
    """
    Resize an image and corresponding annotation and/or label file.

    Args:
        resize_func: A function defining the resize strategy which calls two
        parameters, width and height, and returns the new width and height after resizing.
        eg. resize_func(x, y): return (x/2 if x > 2000 else x, y/3 if y > 3000 else y)
        path_img_folder: Path of the folder containing all the images you wanna resize.
        path_img_out: Path of the folder you wanna save the images after resizing.
        path_anno: Path of the annotation files corresponding to the images.
        path_anno_out: Path of the folder you wanna save the annotation files after resizing.
        path_label_folder: Path of the label files corresponding to the images.
        path_label_out: Path of the folder you wanna save the label files after resizing.

    Returns:
        None
    """
    path_imgs = [os.path.join(path_img_folder, i)
                 for i in os.listdir(path_img_folder)]
    resize_func_ = [resize_func for i in range(len(path_imgs))]
    path_img_out_ = [path_img_out for i in range(len(path_imgs))]

    if path_anno_folder:
        path_annos = [os.path.join(path_anno_folder, i)
                      for i in os.listdir(path_anno_folder)]
    else:
        path_annos = [None for i in range(len(path_imgs))]

    path_anno_out_ = [path_anno_out
                      if path_anno_out else path_anno_out
                      for i in range(len(path_imgs))]

    if path_label_folder:
        path_labels = [os.path.join(path_label_folder, i)
                       for i in os.listdir(path_label_folder)]
    else:
        path_labels = [None for i in range(len(path_imgs))]

    path_label_out_ = [path_label_out
                       if path_label_out else path_label_out
                       for i in range(len(path_imgs))]

    verbose_ = [verbose for i in range(len(path_imgs))]

    pool = Pool(cpu_count())
    pool.starmap(resize_img, zip(resize_func_, path_imgs, path_img_out_,
                                 path_annos, path_anno_out_,
                                 path_labels, path_label_out_, verbose_))
    pool.close()
    pool.join()
