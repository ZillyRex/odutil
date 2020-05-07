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

        {'filename': 'image.jpg',
         'size': {'width': '1621', 'height': '1216', 'depth': '3'},
         'object': [
             {'name': 'class1', 'xmin': '904', 'ymin': '674', 'xmax': '926', 'ymax': '695'},
             {'name': 'class2', 'xmin': '972', 'ymin': '693', 'xmax': '993', 'ymax': '713'}]}
    """
    res = {}
    tree = ET.ElementTree(file=path_anno)
    # Parse filename
    res['filename'] = tree.find('filename').text

    # Parse size
    size = tree.find('size')
    dict_size = {}
    for item in size:
        dict_size[item.tag] = item.text
    res['size'] = dict_size

    # Parse object
    objs = tree.findall('object')
    res['object'] = []
    for obj in objs:
        dict_obj = {}
        dict_obj['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        for item in bbox:
            dict_obj[item.tag] = item.text
        res['object'].append(dict_obj)
    return res


def parse_annos(path_annos):
    """
    Parse a list of annotation files into a list of dicts.

    Args:
        path_anno: The directory of the annotation files you wanna parse.

    Returns:
        A list of dicts. Each of them mapping filename, size and objects in the annotation to the
        corresponding data fetched.
        For example:

        {'filename': 'image.jpg',
         'size': {'width': '1621', 'height': '1216', 'depth': '3'},
         'object': [
             {'name': 'class1', 'xmin': '904', 'ymin': '674', 'xmax': '926', 'ymax': '695'},
             {'name': 'class2', 'xmin': '972', 'ymin': '693', 'xmax': '993', 'ymax': '713'}]}
    """
    path_annos = [os.path.join(path_annos, i) for i in os.listdir(path_annos)]
    pool = Pool(cpu_count())
    res = pool.map(parse_anno, path_annos)
    return res
