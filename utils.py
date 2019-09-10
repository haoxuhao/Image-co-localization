import numpy as np
import xml.etree.ElementTree as ET
import os.path as osp
import os
from prettytable import PrettyTable
import cv2
import selectivesearch
from tqdm import tqdm
import json

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

def load_voc_gts(img_ids, ann_dir):
    '''
    load voc style gts
    param img_ids: list of str
    param ann_dir: ann dir
    return dict: {img_id: np.array([[x1,y1,x2,y2], ..., ])}
    '''
    gts={}
    for id in img_ids:
        xml_file = osp.join(ann_dir, id+".xml")
        gts[id] = np.array([item['bbox'] for item in parse_rec(xml_file)])
    return gts

def gen_voc_imageids(data_root, category):
    if not osp.exists(data_root):
        raise IOError("no such directory: %s"%data_root)

    layout_category = osp.join(data_root,"ImageSets/Main", category+"_trainval.txt")
    ann_dir = osp.join(data_root, "Annotations")
    
    with open(layout_category, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    ret=[]        
    skiped=[]
    for line in lines:
        split_tmp = line.split()
        imageid, tag = split_tmp
        if tag == "1":
            xml_file = osp.join(ann_dir, imageid + ".xml")
            xml_info = parse_rec(xml_file)
            add_flag=False
            for obj in xml_info:
                if obj['name']==category and obj["difficult"] == 0 and obj["truncated"] == 0:
                    add_flag=True
                    break
            if add_flag:
                ret.append(imageid)
            else:
                skiped.append(imageid)

    return ret

def gen_objdis_imageids(image_dir, filter_noise=True):
    imageids = []
    skiped = []
    for file in os.listdir(image_dir):
        if file.find(".jpg") != -1:
            imageid = file.split(".")[0]
            gt_image_path = osp.join(image_dir, "GroundTruth", imageid+".png")
            gt_img = cv2.imread(gt_image_path)
            if np.sum(gt_img) < 1 and filter_noise:
                skiped.append(imageid)
            else:
                imageids.append(imageid)
                
                
    #print("%s: skiped: %d; %s"%(image_dir, len(skiped), str(skiped)))
    return imageids

def results_table_form(results):
    table = PrettyTable(results.keys())
    table.padding_width = 1 # One space between column edges and contents (default)
    table.add_row(results.values())
    return table

def gen_image_proposals(img, min_size=10):
    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=min_size)
    # print("ss time: %.3f s"%(time.time()-start))
    candidates = set()
    min_size_area = min_size*min_size
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < min_size_area:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        #if w / h > 1.2 or h / w > 1.2:
            #continue
        candidates.add(r['rect'])
    return candidates

def load_rois(img_path, filter_wh=(20, 20)):
    if not osp.exists(img_path):
        raise FileExistsError("%s file not found"%img_path)
    pps_path = img_path.replace("100", "100-pps").replace(".jpg", ".json")
    if not osp.exists(pps_path):
        return None
    with open(pps_path) as f:
        pps = json.load(f)["proposals"]
    pps = [p for p in pps if p[2]>filter_wh[0] and p[3]>filter_wh[1]]
    return np.array(pps)

def gen_dataset_proposals(image_paths, output_dir, min_size=10):
    if not osp.exists(output_dir):
        print("output dir not exists, create :%s"%output_dir)
        os.makedirs(output_dir)

    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path)
        img = img[:,:,::-1] #convert BGR to RGB
        proposals = gen_image_proposals(img, min_size=min_size)
        img_filename = osp.basename(img_path)
        save_path = osp.join(output_dir, img_filename.split(".")[0]+".json")
        output_dict = {}
        output_dict["width"]=img.shape[1]
        output_dict["height"]=img.shape[0]
        output_dict["image path"] = img_path
        output_dict["proposals"] = list(proposals) 
        with open(save_path, "w") as f:
            json.dump(output_dict, f)
        
    

if __name__ == "__main__":
    #test one to many iou
    # bb = np.array([20, 30, 50, 50])
    # BBGT = np.array([[20, 30, 50, 50], [25, 35, 65, 50], [0, 0, 0, 0]])
    # iou, idx = iou_one2many(bb, BBGT)
    # print("iou test: ", iou, idx)

    # #test 
    # ret = gen_voc_imageids("./datasets/VOC2007","aeroplane")
    # print(ret)
    # print(len(ret))

    #gen datasets proposals
    objdis_root = "./datasets/ObjectDiscovery-data/Data"
    categories = ["Horse", "Airplane"]
    for category in tqdm(categories):
        images_dir = osp.join(objdis_root, category+"100")
        proposals_save_dir = osp.join(objdis_root, category+"100-pps")

        image_paths = [osp.join(images_dir, file) for file in os.listdir(images_dir)\
            if file.find(".jpg") != -1]

        gen_dataset_proposals(image_paths, proposals_save_dir)
        #print(load_rois(image_paths[0]).shape)
        