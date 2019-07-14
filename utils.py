import numpy as np
import xml.etree.ElementTree as ET
import os.path as osp
import os
from prettytable import PrettyTable
import cv2

'''
borrowed from darknet
'''
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

def iou_one2many(bb, BBGT):
    '''
    borrowed from voc_eval.py
    param bb: [4,]: x1,y1,x2,y2
    param BBGT: [m, 4]: [[x1,y1,x2,y2],...,]
    return: (-1, -1) if m=0 
            else return (maxiou bb_index) 
    '''
    if BBGT.size == 0 or bb.size == 0:
        return -1, -1
    
    # compute overlaps
    # intersection
    ixmin = np.maximum(BBGT[:, 0], bb[0])
    iymin = np.maximum(BBGT[:, 1], bb[1])
    ixmax = np.minimum(BBGT[:, 2], bb[2])
    iymax = np.minimum(BBGT[:, 3], bb[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

    overlaps = inters / uni
    ovmax = np.max(overlaps)
    jmax = np.argmax(overlaps)
    return ovmax, jmax

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(open(filename, encoding="utf-8"))
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text if obj.find('name') is not None else "nknown"
        obj_struct['pose'] = obj.find('pose').text if obj.find('pose') is not None else "nknown"
        obj_struct['truncated'] = int(obj.find('truncated').text) if obj.find('truncated') is not None else 0
        obj_struct['difficult'] = int(obj.find('difficult').text) if obj.find('difficult') is not None else 0
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

def gen_voc_imageids(data_root, category):
    if not osp.exists(data_root):
        raise IOError("no such directory: %s"%data_root)

    layout_category = osp.join(data_root,"ImageSets/Main", category+"_trainval.txt")
    ann_dir = osp.join(data_root, "Annotations")
    
    with open(layout_category, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    ret=[]        
    org_count=0
    skiped=[]
    for line in lines:
        split_tmp = line.split()
        imageid, tag = split_tmp
        if tag == "1":
            xml_file = osp.join(ann_dir, imageid + ".xml")
            xml_info = parse_rec(xml_file)
            org_count+=1
            # print(org_count)
            add_flag=False
            for obj in xml_info:
                if obj['name']==category and obj["difficult"] == 0 and obj["truncated"] == 0:
                    add_flag=True
                    break
            if add_flag:
                ret.append(imageid)
            else:
                skiped.append(imageid)


    #print("skip: %s"%str(skiped))
    return ret

def gen_objdis_imageids(image_dir, filter_noise=True):
    imageids = []
    skiped = []
    for file in os.listdir(image_dir):
        if file.find(".jpg") != -1:
            imageid = file.split(".")[0]
            gt_image_path = osp.join(image_dir, "GroundTruth", imageid+".png")
            gt_img = cv2.imread(gt_image_path)
            if np.sum(gt_img) > 1:
                imageids.append(imageid)
            else:
                skiped.append(imageid)
                
    print("%s: skiped: %d; %s"%(image_dir, len(skiped), str(skiped)))
    return imageids

def results_table_form(results):
    table = PrettyTable(results.keys())
    table.padding_width = 1 # One space between column edges and contents (default)
    table.add_row(results.values())
    #print(table)
    return table

if __name__ == "__main__":
    #test one to many iou
    bb = np.array([20, 30, 50, 50])
    BBGT = np.array([[20, 30, 50, 50], [25, 35, 65, 50], [0, 0, 0, 0]])
    iou, idx = iou_one2many(bb, BBGT)
    print("iou test: ", iou, idx)

    #test 
    ret = gen_voc_imageids("./datasets/VOC2007","aeroplane")
    print(ret)
    print(len(ret))