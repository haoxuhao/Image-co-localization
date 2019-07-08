#!/usr/bin/env python
#encoding=utf-8

import os
import os.path as osp
from utils import iou_one2many, parse_rec, gen_voc_imageids, gen_objdis_imageids
import numpy as np
from config import Dataset
import argparse


def load_voc_gts(img_ids, ann_dir):
    '''
    load voc style gts
    param img_ids: list of str
    param ann_dir: ann dir
    return dict: {img_id: np.array([[x1,y1,x2,y2], ..., ])}
    '''
    print("load voc gts.")
    gts={}
    for id in img_ids:
        xml_file = osp.join(ann_dir, id+".xml")
        gts[id] = np.array([item['bbox'] for item in parse_rec(xml_file)])
    return gts


def eval_corloc(result_file, gts, iou_thresh=0.5):
    if not osp.exists(result_file):
        raise IOError("no such file: %s"%result_file)
    with open(result_file,"r") as f:
        preds = [line.strip("\n") for line in f.readlines()]

    results = np.zeros((len(preds),))

    for i, pred in enumerate(preds):
        line_splits = pred.split(" ")
        img_id = line_splits[0]
        if len(line_splits) == 1:
            if gts[img_id].size == 0:
                results[i]=1
        else:
            bbox = np.array([int(line_splits[1]), int(line_splits[2]), int(line_splits[3]), int(line_splits[4])])
            iou, idx = iou_one2many(bbox, gts[img_id])
            if iou > iou_thresh:
                results[i]=1

    corloc = np.sum(results) / float(len(preds))

    return corloc
                
def voc_eval(category, dataset_type):
    val_image_ids = gen_voc_imageids("./datasets/VOC2007",category)
    print("len of val image ids: %d"%len(val_image_ids))

    gts = load_voc_gts(val_image_ids, "./datasets/VOC2007/Annotations")
    result_file = "./data/result/%s-%s/result.txt"%(dataset_type, category)
    corloc = eval_corloc(result_file, gts)

    print("corloc: %.3f"%corloc)

def objdis_eval(category, dataset_type):
    val_image_ids = gen_objdis_imageids("./datasets/ObjectDiscovery-data/Data/%s100"%category)
    print("len of val image ids: %d"%len(val_image_ids))

    gts = load_voc_gts(val_image_ids, "./datasets/ObjectDiscovery-data/Data/%s100-bbox-ann"%category)
    result_file = "./data/result/%s-%s100/result.txt"%(dataset_type,category)
    corloc = eval_corloc(result_file, gts)

    print("corloc: %.3f"%corloc)


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", "-dset_type", type=str, default="voc07", help="dataset type")
    parser.add_argument("--category", "-cate", type=str, default="aeroplane", help="category")

    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    args.category = "aeroplane" 
    args.dataset_type = "voc07"
    print("dataset type: %s"%args.dataset_type)
    print("category: %s"%args.category)

    if args.dataset_type == Dataset.voc07:
        if not args.category in Dataset.voc_classes:
            raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
        voc_eval(args.category, args.dataset_type)
    elif args.dataset_type == Dataset.objdis:
        if not args.category in Dataset.objdis_classes:
            raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
        objdis_eval(args.category, args.dataset_type)
    else:
        raise Exception("no such dataset type: %s" % args.dataset_type)

if __name__ == "__main__":
    main()

    
    

    
    