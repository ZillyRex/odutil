# -*- coding: utf-8 -*-
# @Author  : ZillyRex


import os
import xml.etree.ElementTree as ET
import json
from multiprocessing import Pool, cpu_count
import pickle
import numpy as np
from tqdm import tqdm


def parse_anno(path_anno, verbose=0):
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
         'objects': [
             {'name': 'class1', 'xmin': '904', 'ymin': '674',
                 'xmax': '926', 'ymax': '695'},
             {'name': 'class2', 'xmin': '972', 'ymin': '693', 'xmax': '993', 'ymax': '713'}]}
    """
    if verbose:
        print(path_anno)
    res = {}
    tree = ET.ElementTree(file=path_anno)

    # Parse annotation name
    res['annoname'] = os.path.basename(path_anno)

    # Parse size
    size = tree.find('size')
    dict_size = {}
    for item in size:
        dict_size[item.tag] = int(float(item.text))
    res['size'] = dict_size

    # Parse object
    objs = tree.findall('object')
    res['objects'] = []
    for obj in objs:
        dict_obj = {}
        dict_obj['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        for item in bbox:
            dict_obj[item.tag] = int(float(item.text))
        res['objects'].append(dict_obj)
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
    for bbox in anno['objects']:
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


def bbox_dist(path_anno_folder, dataset=None, verbose=0):
    """
    Analysis the bbox distribution by a list of annotation files.

    Args:
        path_anno_folder: Path of the annotation files folder.

    Returns:
        A dict contains the result.
    """
    annos = parse_annos(path_anno_folder)
    d = {}
    d['LEN'] = 0
    if dataset:
        base_names = set()
        with open(dataset) as f:
            for l in f:
                base_names.add(os.path.splitext(
                    os.path.basename(l.strip()))[0])
    for annoname in annos:
        if dataset and (annoname not in base_names):
            continue
        anno = annos[annoname]
        objs = anno['objects']
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
        name2label, label2name: A dict like {'name1': 0, 'name2': 1, ...},
                                A dict like {0: 'name1', 1: 'name2', ...}
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


def base2abs(path_base, path_prefix, path_out):
    """
    Convert the base names to absolute paths.

    Args:
        path_base: Path of the base file.
        path_prefix: Prefix path you wanna put behind the base names.
        path_out: Path of the output file.

    Returns:
        None
    """
    prefix_abs = os.path.abspath(path_prefix)
    bases = []
    with open(path_base) as f:
        for line in f:
            bases.append(line.strip('\n'))

    bases_abs = [os.path.join(prefix_abs, i) for i in bases]
    with open(path_out, 'w') as f:
        f.write('\n'.join(bases_abs))


def split_trainval(path_img_folder, path_train, path_val, ratio_train):
    '''
    Split the date into training & validation subset by basename format.

    Param:
        path_img_folder: Path of JPEImages.
        path_train: Path of the txt file containing training subset.
        path_val: Path of the txt file containing validation subset.
        ratio_train: The ratio of training set which should be more than 0 and less or equal than 1.

    Return:
        None
    '''
    if not 0 < ratio_train <= 1:
        print('Please set a right ratio_train.')
        return
    name_all = os.listdir(path_img_folder)
    np.random.seed(0)
    np.random.shuffle(name_all)
    len_train = int(len(name_all)*ratio_train)
    name_train = name_all[:len_train]
    name_test = name_all[len_train:]
    with open(path_train, 'w') as f:
        f.write('\n'.join(name_train))
    with open(path_val, 'w') as f:
        f.write('\n'.join(name_test))


def iou(bbox_1, bbox_2):
    """
    bbox_i: (xmin, ymin, xmax, ymax)
    """
    inter_xmin = max(bbox_1[0], bbox_2[0])
    inter_xmax = min(bbox_1[2], bbox_2[2])
    inter_ymin = max(bbox_1[1], bbox_2[1])
    inter_ymax = min(bbox_1[3], bbox_2[3])
    if(inter_xmin >= inter_xmax or inter_ymin >= inter_ymax):
        return 0
    inter_area = (inter_xmax-inter_xmin)*(inter_ymax-inter_ymin)
    union_area = (bbox_1[2]-bbox_1[0])*(bbox_1[3]-bbox_1[1]) + \
        (bbox_2[2]-bbox_2[0])*(bbox_2[3]-bbox_2[1])-inter_area
    return inter_area/union_area


def tp_fp_fn(ground_truth, results, conf_thresh, iou_thresh):
    """
    tp, fp, fn for each class
    """

    tp_fp_fn_res = {}
    for base_name in results:
        objs_pre = results[base_name]
        objs_pre.sort(key=lambda x: x['conf'], reverse=True)

        objs_gt = ground_truth[base_name]['objects']
        occupied = [0 for i in range(len(objs_gt))]

        for obj_pre in objs_pre:
            if obj_pre['name'] not in tp_fp_fn_res:
                tp_fp_fn_res[obj_pre['name']] = {'tp': 0, 'fp': 0, 'fn': 0}
            if obj_pre['conf'] < conf_thresh:
                continue
            cls_gt = ''
            iou_score = 0
            id_gt = -1
            for i, obj_gt in enumerate(objs_gt, 0):
                if obj_gt['name'] not in tp_fp_fn_res:
                    tp_fp_fn_res[obj_gt['name']] = {'tp': 0, 'fp': 0, 'fn': 0}
                if occupied[i] != 0:
                    continue
                cur_iou = iou((obj_gt['xmin'], obj_gt['ymin'], obj_gt['xmax'], obj_gt['ymax']),
                              (obj_pre['xmin'], obj_pre['ymin'], obj_pre['xmax'], obj_pre['ymax']))
                if cur_iou > iou_score:
                    iou_score = cur_iou
                    cls_gt = obj_gt['name']
                    id_gt = i
            if iou_score > iou_thresh and obj_pre['name'] == cls_gt:
                tp_fp_fn_res[obj_pre['name']]['tp'] += 1
                occupied[id_gt] = 1
            else:
                tp_fp_fn_res[obj_pre['name']]['fp'] += 1

        for i in occupied:
            if i == 0:
                if objs_gt[i]['name'] not in tp_fp_fn_res:
                    tp_fp_fn_res[objs_gt[i]['name']] = {
                        'tp': 0, 'fp': 0, 'fn': 0}
                tp_fp_fn_res[objs_gt[i]['name']]['fn'] += 1
    return tp_fp_fn_res


def _convert_darknet(dir_gt, path_res, mode):
    """
    mode: gt {0, 1}: annotations, labels(cls_id cx cy w h)
          res {0, 1, 2}: .json by test, comp4_ by valid, .txt by test output
    """
    ground_truth, results = {}, {}
    mode = str(mode)

    if mode[1] == '0':
        # Annotations
        ground_truth = parse_annos(dir_gt)
    elif mode[1] == '1':
        # labels
        pass
    else:
        raise Exception('Invalid mode.')

    if mode[2] == '0':
        # .json
        with open(path_res) as f:
            jsonData = f.readlines()
            jsonData = ''.join(jsonData)
            results_data = json.loads(jsonData)
        for frame in results_data:
            base_name = os.path.splitext(
                os.path.basename(frame['filename']))[0]
            W = ground_truth[base_name]['size']['width']
            H = ground_truth[base_name]['size']['height']
            objs = []
            for obj in frame['objects']:
                cx = obj['relative_coordinates']['center_x']
                cy = obj['relative_coordinates']['center_y']
                w = obj['relative_coordinates']['width']
                h = obj['relative_coordinates']['height']
                xmin = int((cx-w/2)*W)
                ymin = int((cy-h/2)*H)
                xmax = int((cx+w/2)*W)
                ymax = int((cy+h/2)*H)
                objs.append({'name': obj['name'],
                             'xmin': xmin,
                             'ymin': ymin,
                             'xmax': xmax,
                             'ymax': ymax,
                             'conf': obj['confidence']})
            results[base_name] = objs
    elif mode[2] == '1':
        # .txt
        pass
    elif mode[2] == '2':
        # comp4_
        pass
    else:
        raise Exception('Invalid mode.')

    return ground_truth, results


def _convert_mmdet(dir_gt, path_res, mode, dataset, names):
    """
    mode: gt {0, 1}: annotations, labels(cls_id cx cy w h)
          res {0}: .pkl by test
    """

    ground_truth, results = {}, {}
    _, label2name = get_label(names)
    mode = str(mode)

    if mode[1] == '0':
        # Annotations
        ground_truth = parse_annos(dir_gt)
    elif mode[1] == '1':
        # labels
        pass
    else:
        raise Exception('Invalid mode.')

    if mode[2] == '0':
        # .pkl or .pickle
        with open(path_res, 'rb') as f:
            results_data = pickle.load(f, encoding='utf-8')

        with open(dataset) as f:
            base_names = []
            for line in f:
                base_names.append(os.path.splitext(
                    os.path.basename(line.strip()))[0])

        for frame_id, base_name in enumerate(base_names, 0):
            objs = []
            for cls_id, objs_pre in enumerate(results_data[frame_id]):
                for obj_pre in objs_pre:
                    objs.append({'name': label2name[cls_id],
                                 'xmin': obj_pre[0],
                                 'ymin': obj_pre[1],
                                 'xmax': obj_pre[2],
                                 'ymax': obj_pre[3],
                                 'conf': obj_pre[4]})
                results[base_name] = objs
    else:
        raise Exception('Invalid mode.')

    return ground_truth, results


def _convert_simpledet(dir_gt, path_res, mode, dataset, names):
    """
    mode: gt {0, 1}: annotations, labels(cls_id cx cy w h)
          res {0}: .json by test
    """

    ground_truth, results = {}, {}
    _, label2name = get_label(names)
    mode = str(mode)

    if mode[1] == '0':
        # Annotations
        ground_truth = parse_annos(dir_gt)
    elif mode[1] == '1':
        # labels
        pass
    else:
        raise Exception('Invalid mode.')

    if mode[2] == '0':
        # .json
        with open(path_res) as f:
            jsonData = f.readlines()
            jsonData = ''.join(jsonData)
            results_data = json.loads(jsonData)

        with open(dataset) as f:
            base_names = {}
            images_id = 0
            for line in f:
                base_names[images_id] = os.path.splitext(
                    os.path.basename(line.strip()))[0]
                images_id += 1

        for box in results_data:
            base_name = base_names[box['image_id']]
            if base_name not in results:
                results[base_name] = []
            x, y, w, h = box['bbox']
            # W, H = ground_truth[base_name]['size']['width'], ground_truth[base_name]['size']['height']
            # x = W*x/800
            # y = H*y/1200
            # w = W*w/800
            # h = H*h/1200
            # TODO: SimpleDet Result ERROR！！！
            results[base_name].append({'name': label2name[box['category_id']],
                                       'xmin': x,
                                       'ymin': y,
                                       'xmax': x+w,
                                       'ymax': y+h,
                                       'conf': box['score']})
    else:
        raise Exception('Invalid mode.')

    return ground_truth, results


def _convert_yolov5(dir_gt, path_res, mode, names):
    """
    mode: gt {0, 1}: annotations, labels(cls_id cx cy w h)
          res {0}: .json by test
    """

    ground_truth, results = {}, {}
    _, label2name = get_label(names)
    mode = str(mode)

    if mode[1] == '0':
        # Annotations
        ground_truth = parse_annos(dir_gt)
    elif mode[1] == '1':
        # labels
        pass
    else:
        raise Exception('Invalid mode.')

    if mode[2] == '0':
        # .json
        with open(path_res) as f:
            jsonData = f.readlines()
            jsonData = ''.join(jsonData)
            results_data = json.loads(jsonData)

        for box in results_data:
            base_name = box['image_id']
            if base_name not in results:
                results[base_name] = []
            x, y, w, h = box['bbox']
            results[base_name].append({'name': label2name[box['category_id']-1],
                                       'xmin': x,
                                       'ymin': y,
                                       'xmax': x+w,
                                       'ymax': y+h,
                                       'conf': box['score']})
    else:
        raise Exception('Invalid mode.')

    return ground_truth, results


def convert_gt_results(gt, res, mode, dataset=None, names=None):
    """
    mode: 1xx: darknet, 2xx: mmdet, 3xx: simpledet, 4xx: yolov5
    final gt format: Same as parse_anno
    final res format: {base_name(no extend): [{'name':str, 'xmin':dig, 'ymin':dig, 'xmax':dig, 'ymax':dig, 'conf':dig}]}
    """

    mode = str(mode)
    if mode[0] == '1':
        return _convert_darknet(gt, res, mode)
    if mode[0] == '2':
        if not (dataset and names):
            raise Exception(
                'dataset and names must be set if the mode is 2xx.')
        return _convert_mmdet(gt, res, mode, dataset, names)
    if mode[0] == '3':
        if not (dataset and names):
            raise Exception(
                'dataset and names must be set if the mode is 3xx.')
        return _convert_simpledet(gt, res, mode, dataset, names)
    if mode[0] == '4':
        if not names:
            raise Exception(
                'names must be set if the mode is 4xx.')
        return _convert_yolov5(gt, res, mode, names)

    raise Exception('Invalid mode.')


def precision_recall(ground_truth, results, conf_thresh, iou_thresh=0.5, verbose=0):
    """
    precision recall
    """
    tp_fp_fn_res = tp_fp_fn(ground_truth, results, conf_thresh, iou_thresh)

    for clss in tp_fp_fn_res:
        tp = tp_fp_fn_res[clss]['tp']
        fp = tp_fp_fn_res[clss]['fp']
        fn = tp_fp_fn_res[clss]['fn']
        precision = np.round(tp/(tp+fp+1e-8), 5)
        recall = np.round(tp/(tp+fn+1e-8), 5)
        tp_fp_fn_res[clss]['precision'] = precision
        tp_fp_fn_res[clss]['recall'] = recall
        if verbose:
            print('{:12}: TP:{:5}, FP:{:5}, FN:{:5}, Precision: {}, Recall: {}'.format(
                clss,
                tp_fp_fn_res[clss]['tp'],
                tp_fp_fn_res[clss]['fp'],
                tp_fp_fn_res[clss]['fn'],
                tp_fp_fn_res[clss]['precision'],
                tp_fp_fn_res[clss]['recall']))

    return tp_fp_fn_res


def mAP(gt, res, mode, dataset=None, names=None, verbose=0, draw=False):
    """
    mAP
    """
    AP = {}
    pbar = tqdm(total=len(np.arange(0, 1.01, 0.01)),
                ncols=100, desc='Processing mAP: ')
    ground_truth, results = convert_gt_results(
        gt, res, mode, dataset=dataset, names=names)
    for thresh in np.arange(0, 1.01, 0.01):
        tp_fp_fn_res = precision_recall(ground_truth, results, thresh)
        for clss in tp_fp_fn_res:
            if clss not in AP:
                AP[clss] = {'precision': [], 'recall': []}
            AP[clss]['precision'].append(tp_fp_fn_res[clss]['precision'])
            AP[clss]['recall'].append(tp_fp_fn_res[clss]['recall'])
        pbar.update()
    pbar.close()

    mAP_res = []
    for clss in AP:
        mAP_res.append(np.mean(AP[clss]['precision']))
    mAP_res = np.mean(mAP_res)

    if verbose:
        for clss in AP:
            print('{:12}: AP: {:.4f}, AR: {:.4f}'.format(
                clss, np.mean(AP[clss]['precision']), np.mean(AP[clss]['recall'])))
        print('mAP: {:.4f}'.format(mAP_res))

    if draw:
        pass

    return mAP_res
