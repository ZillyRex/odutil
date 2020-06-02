# -*- coding: utf-8 -*-
# @Author  : ZillyRex


import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count


def parse_anno(path_anno):
    """
    Parse an annotation file into a dict.

    Args:
        path_anno: The path of the annotation file you wanna parse.

    Returns:
        A dict mapping filename, size and objects in the annotation to the
        corresponding data fetched.
        For example:

        {'annoname': 'xxx.xml',
         'filename': 'image.jpg',
         'size': {'width': '1621', 'height': '1216', 'depth': '3'},
         'object': [
             {'name': 'class1', 'xmin': '904', 'ymin': '674', 'xmax': '926', 'ymax': '695'},
             {'name': 'class2', 'xmin': '972', 'ymin': '693', 'xmax': '993', 'ymax': '713'}]}
    """
    res = {}
    tree = ET.ElementTree(file=path_anno)

    # Parse annotation name
    res['annoname'] = os.path.basename(path_anno)

    # Parse filename
    res['filename'] = tree.find('filename').text

    # Parse size
    size = tree.find('size')
    dict_size = {}
    for item in size:
        dict_size[item.tag] = int(float(item.text))
    res['size'] = dict_size

    # Parse object
    objs = tree.findall('object')
    res['object'] = []
    for obj in objs:
        dict_obj = {}
        dict_obj['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        for item in bbox:
            dict_obj[item.tag] = int(float(item.text))
        res['object'].append(dict_obj)
    return res


def parse_annos(path_anno_folder):
    """
    Parse a list of annotation files into a list of dicts.

    Args:
        path_anno_folder: Path of the directory of the annotation files you wanna parse.

    Returns:
        A dict of dicts. Each of them mapping annotation file name("annoname")
        to the corresponding annotation dict fetched by parse_anno().
    """
    path_annos = [os.path.join(path_anno_folder, i)
                  for i in os.listdir(path_anno_folder)]
    pool = Pool(cpu_count())
    res = pool.map(parse_anno, path_annos)
    pool.close()
    pool.join()
    annos_dict = {}
    for anno in res:
        annos_dict[os.path.splitext(anno['annoname'])[0]] = anno
    return annos_dict


def is_match(path_1, path_2):
    """
    Check if the file names in the folders match each other.

    Param:
        path_1: Path of a folder.
        path_2: Path of another folder.

    Return:
        True if match else False.
    """
    name_1 = os.listdir(path_1)
    name_2 = os.listdir(path_2)
    if len(name_1) != len(name_2):
        return False
    set_name = set()
    for name in name_1:
        set_name.add(os.path.splitext(name)[0])
    for name in name_2:
        if os.path.splitext(name)[0] not in set_name:
            return False
    return True


def anno2label(path_anno, path_names, path_out):
    """
    Generate label txt from annotation.

    Args:
        path_anno: Path of the annotation file.
        path_names: Path of the .names file. Only the class in the .names will be converted.
        path_out: Path of the output .txt file.

    Returns:
        None
    """
    anno = parse_anno(path_anno)
    name2label, _ = get_label(path_names)

    W, H = anno['size']['width'], anno['size']['height']
    row = []
    for bbox in anno['object']:
        if bbox['name'] not in name2label:
            continue
        label = name2label[bbox['name']]
        x_center = (bbox['xmin']+bbox['xmax'])/(2*W)
        y_center = (bbox['ymin']+bbox['ymax'])/(2*H)
        width = (bbox['xmax']-bbox['xmin'])/W
        height = (bbox['ymax']-bbox['ymin'])/H
        row.append(
            ' '.join(list(map(str, [label, x_center, y_center, width, height]))))

    if not os.path.isdir(path_out):
        os.mkdir(path_out)
    file_name = os.path.splitext(os.path.basename(path_anno))[0]
    path_out_file = os.path.join(path_out, '{}.txt'.format(file_name))
    with open(path_out_file, 'w') as f:
        f.write('\n'.join(row))


def annos2labels(path_anno_folder, path_names, path_out):
    """
    Generate label txt from a list of annotations.

    Args:
        path_anno_folder: Path of the annotation files folder.
        path_names: Path of the .names file.
        path_out: Path of the output .txt file.

    Returns:
        None
    """
    path_annos = [os.path.join(path_anno_folder, i)
                  for i in os.listdir(path_anno_folder)]
    path_names_ = [path_names for i in range(len(path_annos))]
    path_out_ = [path_out for i in range(len(path_annos))]
    pool = Pool(cpu_count())
    pool.starmap(anno2label, zip(path_annos, path_names_,
                                 path_out_,))
    pool.close()
    pool.join()


def bbox_dist(path_anno_folder, verbose=0):
    """
    Analysis the bbox distribution by a list of annotation files.

    Args:
        path_anno_folder: Path of the annotation files folder.

    Returns：
        A dict contains the result.
    """
    annos = parse_annos(path_anno_folder)
    d = {}
    d['LEN'] = 0
    for annoname in annos:
        anno = annos[annoname]
        objs = anno['object']
        for obj in objs:
            d[obj['name']] = d.setdefault(obj['name'], 0)+1
            d['LEN'] += 1
    if verbose:
        for name in d:
            print('{}: {}({:.4f}%)'.format(
                name, d[name], 100*d[name]/d['LEN']))
    return d


def get_label(path_names):
    """
    Get name2label and label2name from a .names file.

    Args:
        path_names: Path of the .names file.

    Returns:
        name2label: A dict like {'name1': 0, 'name2': 1, ...}
        label2name: A dict like {0: 'name1', 1: 'name2', ...}
    """
    name2label = {}
    label2name = {}
    with open(path_names) as f:
        label = 0
        for l in f:
            name = l.strip('\n')
            name2label[name] = label
            label2name[label] = name
            label += 1
    return name2label, label2name
