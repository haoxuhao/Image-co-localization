#encoding=utf-8

import os
import os.path as osp
import sys
sys.path.append("..")
from utils import *

def calc_recalls(category, iou_thresh=0.5):
    imagedir = "../datasets/ObjectDiscovery-data/Data/%s100"%category
    val_image_ids = gen_objdis_imageids(imagedir)
    gts = load_voc_gts(val_image_ids, "../datasets/ObjectDiscovery-data/Data/%s100-bbox-ann"%category)
    total = 0
    tp = 0
    for imageid in val_image_ids:
        image_path = osp.join(imagedir, imageid+".jpg")
        pps = load_rois(image_path)
        pps[:, 2] = pps[:, 0] + pps[:, 2]
        pps[:, 3] = pps[:, 1] + pps[:, 3]
        gt = gts[imageid]
        total += gt.shape[0]

        for i in range(gt.shape[0]):
            iou, idx = iou_one2many(gt[i,:], pps)
            if iou > iou_thresh:
                tp += 1
                
    recall = tp / total

    return recall

if __name__ == "__main__":
    category = "Car"
    ret = calc_recalls(category)
    print("%s recalls: %.3f"%(category, ret))

    category = "Horse"
    ret = calc_recalls(category)
    print("%s recalls: %.3f"%(category, ret))

    category = "Airplane"
    ret = calc_recalls(category)
    print("%s recalls: %.3f"%(category, ret))



        
        
