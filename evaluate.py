#!/usr/bin/env python
#encoding=utf-8

import os
import os.path as osp
from utils import iou_one2many, parse_rec, gen_voc_imageids, gen_objdis_imageids,results_table_form,load_voc_gts
import numpy as np
from config import Dataset
import argparse
from tqdm import tqdm


def eval_corloc(result_file, gts, iou_thresh=0.5):
    if not osp.exists(result_file):
        raise IOError("no such file: %s"%result_file)
    with open(result_file,"r") as f:
        preds = [line.strip("\n") for line in f.readlines()]
    if len(preds)==0:
        print("error: no dets been found")
        return 0
    results = np.zeros((len(preds),))
    error_file = open(osp.join(osp.dirname(result_file), "error.txt"), "w")
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
            else:
                error_file.write(img_id+"\n")
    error_file.close()

    corloc = np.sum(results) / float(len(preds))

    return corloc
                
def voc_eval(category, dataset_type, dataset_root="./datasets/VOC2007"):
    val_image_ids = gen_voc_imageids(dataset_root,category)

    gts = load_voc_gts(val_image_ids, osp.join(dataset_root, "Annotations"))
    result_file = "./data/result/%s-%s/result.txt"%(dataset_type, category)
    corloc = eval_corloc(result_file, gts)

    return corloc

def objdis_eval(category, dataset_type):
    val_image_ids = gen_objdis_imageids("./datasets/ObjectDiscovery-data/Data/%s100"%category)

    gts = load_voc_gts(val_image_ids, "./datasets/ObjectDiscovery-data/Data/%s100-bbox-ann"%category)
    result_file = "./data/result/%s-%s100/result.txt"%(dataset_type,category)
    corloc = eval_corloc(result_file, gts)

    return corloc


def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_type", "-dset_type", type=str, default="objdis", help="dataset type")
    parser.add_argument("--category", "-cate", type=str, default=None, help="category")

    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    if args.category and args.dataset_type in [Dataset.objdis, Dataset.voc07, Dataset.voc12]:
        args.categories=[args.category]
    elif args.dataset_type == Dataset.objdis:
        args.categories = Dataset.objdis_classes
    elif args.dataset_type == Dataset.voc07:
        args.categories = Dataset.voc_classes#["Airplane"]
    elif args.dataset_type == Dataset.voc12:
        args.categories = Dataset.voc_classes 
    else:
        raise Exception("unkonwn dataset type: %s"%args.dataset_type)

    print("dataset: %s"%args.dataset_type)

    result_all = {}
    result_avg = 0
    classes = len(args.categories)
    for category in tqdm(args.categories):
        args.category = category
        if args.dataset_type == Dataset.voc07:
            if not args.category in Dataset.voc_classes:
                raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
            ret=voc_eval(args.category, args.dataset_type)
            result_avg+=ret
            result_all[category] = "%.1f"%(ret*100)
        elif args.dataset_type == Dataset.objdis:
            if not args.category in Dataset.objdis_classes:
                raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
            ret=objdis_eval(args.category, args.dataset_type)
            result_avg+=ret
            result_all[category] = "%.1f"%(ret*100)
        elif args.dataset_type == Dataset.voc12:
            if not args.category in Dataset.voc_classes:
                raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
            ret=voc_eval(args.category, args.dataset_type, dataset_root="./datasets/VOC2012")
            result_avg+=ret
            result_all[category] = "%.1f"%(ret*100)
        else:
            raise Exception("no such dataset type: %s" % args.dataset_type)

    result_all["Mean"] = "%.1f"%(100*result_avg/classes)
    print(results_table_form(result_all))
    

if __name__ == "__main__":
    main()

    
    

    
    
