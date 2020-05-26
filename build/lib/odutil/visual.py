# -*- coding: utf-8 -*-
# @Author  : ZillyRex


import os
from multiprocessing import Pool, cpu_count
import cv2.cv2 as cv2
from . import analysis


def draw_bbox(img, *args):
    """
    Draw the bounding box on an image.

    Args:
        img: An image instance obtained by cv2.imread.
        args: {cls_, x, y, w, h, color_bbox, color_text}.
            cls_ is object class name.
            x, y are the coordinates of the bbox top-left corner.
            w, h are the weight and height of the bbox.
            color_bbox is the (B, G, R) color for bbox.
            color_bbox is the (B, G, R) color for text.

    Returns:
        The img with bbox on it.
    """
    cls_, x, y, w, h, color_bbox, color_text = args
    cv2.rectangle(img, (x, y), (x+w, y+h), color_bbox, 4)
    cv2.putText(img, cls_, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color_text, 1)
    return img


def draw_anno(path_img, path_anno, path_out, verbose=0):
    """
    Draw the bounding box of on an image by its annotation file.

    Args:
        path_img: Path of the image.
        path_anno: Path of the corresponding annotation file.
        path_out: Path of the output folder.

    Returns:
        None
    """
    img = cv2.imread(path_img)
    anno = analysis.parse_anno(path_anno)
    color_bbox = (0, 255, 0)
    color_text = (0, 0, 255)
    params = []
    for obj in anno['object']:
        params.append((obj['name'],
                       obj['xmin'], obj['ymin'],
                       obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin'],
                       color_bbox, color_text))
    for param in params:
        draw_bbox(img, *param)
    cv2.imwrite(os.path.join(path_out, '{}'.format(
        os.path.basename(path_img))), img)
    if verbose:
        print(path_img)


def draw_annos(path_img_folder, path_anno_folder, path_out, verbose=0):
    """
    Draw the bounding box of on a list of images by their annotation files.

    Args:
        path_img_folder: Path of the images folder.
        path_anno_folder: Path of the corresponding annotation folder.
        path_out: Path of the output folder.

    Returns:
        None
    """
    if not analysis.check_match(path_anno_folder, path_img_folder):
        print('file names in the images and annotation folders don\'t match each other!')
        return
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    path_imgs = [os.path.join(path_img_folder, i)
                 for i in os.listdir(path_img_folder)]
    path_annos = [os.path.join(path_anno_folder, i)
                  for i in os.listdir(path_anno_folder)]
    path_out = [path_out for i in range(len(path_imgs))]
    verbose_ = [verbose for i in range(len(path_imgs))]
    pool = Pool(cpu_count())
    pool.starmap(draw_anno, zip(path_imgs, path_annos, path_out, verbose_))
    pool.close()
    pool.join()


def draw_result(path_result, path_img_folder, path_out, verbose=0):
    """
    Draw the bounding box by a result file.
    The result file format is as follow:
        ...... /xxx/xxx/xxx.jpg: ......
        ...... /xxx/xxx/xxx.jpg: ......
        class_name: 75%	(left_x:  626   top_y:  961   width:   50   height:   18)
        class_name: 84%	(left_x:  626   top_y:  943   width:   47   height:   20)

    Args:
        path_result: Path of the result file.
        path_img_folder: Path of the folder containing all the images in the result file.
        path_out: Path of the output folder.

    Returns:
        None
    """
    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    args_list = []
    basename = None
    with open(path_result) as f:
        for line in f:
            if '/' in line:
                if args_list:
                    img = cv2.imread(os.path.join(path_img_folder, basename))
                    for args in args_list:
                        draw_bbox(img, *args)
                    cv2.imwrite(os.path.join(path_out, basename), img)
                    args_list.clear()
                    if verbose:
                        print(basename)
                abspth = line.split()[3][:-1]
                basename = os.path.basename(abspth)
            if '%' in line:
                l = line.strip('\n').split()
                cls_ = l[0][:-1]
                x = int(float(l[3]))
                y = int(float(l[5]))
                w = int(float(l[7]))
                h = int(float(l[9][:-1]))
                args_list.append(
                    (cls_, x, y, w, h, (0, 255, 255), (0, 255, 255)))
