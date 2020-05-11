import os
from multiprocessing import Pool, cpu_count
import cv2.cv2 as cv2
from . import analysis


def draw_bbox(path_img, path_anno, path_out):
    """
    Draw the bounding box of on an image by its annotation file.

    Args:
        path_img: Path of the image.
        path_anno: Path of the corresponding annotation file.
        path_out: Path of the output folder.
        color_bbox: Color of the box.(B, G, R)
        color_text: Color of the text.(B, G, R)

    Return:
        None
    """
    img = cv2.imread(path_img)
    anno = analysis.parse_anno(path_anno)
    params = []
    for obj in anno['object']:
        params.append([obj['name'], obj['xmin'],
                       obj['ymin'], obj['xmax'], obj['ymax']])
    color_bbox = (0, 255, 0)
    color_text = (0, 0, 255)
    for param in params:
        cls, xmin, ymin, xmax, ymax = param
        # BBOX
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color_bbox, 4)
        # TEXT
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, cls, (xmin, ymin-10), font, 2, color_text, 2)
    cv2.imwrite(os.path.join(path_out, 'bbox_{}'.format(
        os.path.basename(path_img))), img)


def draw_bboxs(path_img_folder, path_anno_folder, path_out):
    """
    Draw the bounding box of on a list of images by their annotation files.

    Args:
        path_img_folder: Path of the images folder.
        path_anno_folder: Path of the corresponding annotation folder.
        path_out: Path of the output folder.
        color_bbox: Color of the box. (B, G, R)
        color_text: Color of the text. (B, G, R)

    Return:
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
    color_bbox = (0, 255, 0)
    color_text = (0, 0, 255)
    color_bbox_ = [color_bbox for i in range(len(path_imgs))]
    color_text_ = [color_text for i in range(len(path_imgs))]
    pool = Pool(cpu_count())
    pool.starmap(draw_bbox, zip(path_imgs, path_annos,
                                path_out, color_bbox_, color_text_))
    pool.close()
    pool.join()
