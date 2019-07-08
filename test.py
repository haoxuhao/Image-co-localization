#!/usr/bin/env python

from DDT import *
from DDTplus import *
import os.path as osp
import os
from utils import gen_voc_imageids, gen_objdis_imageids
from config import Dataset
import argparse


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, default='./data/car', help="the train data folder path.")
    parser.add_argument('--testdir', type=str, default='./data/car', help="the test data folder path.")
    parser.add_argument('--savedir', type=str, default='./data/result/car', help="the final result saving path. ")
    parser.add_argument('--select_layers', type=tuple, default=(34, 36), help="two selected layers index. Note that the output channel should be 512. ")
    parser.add_argument("--gpu", type=str, default="0", help="cuda device to run")
    parser.add_argument("--algorithm", "-a", type=str, default="ddt", help="algorithm to use")
    args = parser.parse_args()
    return args

def main():
    args = parse_arg()
    args.dataset_type = "voc07"
    category="aeroplane" #"aeroplane"

    if args.dataset_type == Dataset.voc07:
        if not category in Dataset.voc_classes:
            raise Exception("no such category: %s in dataset %s"%(category, args.dataset_type)) 
        args.traindir = "./datasets/VOC2007/JPEGImages"
        args.testdir = "./datasets/VOC2007/JPEGImages"
        args.savedir = "./data/result/%s-%s"%(args.dataset_type, category)
        val_image_ids = gen_voc_imageids("./datasets/VOC2007", category)
    elif args.dataset_type == Dataset.objdis:
        if not category in Dataset.objdis_classes:
            raise Exception("no such category: %s in dataset %s"%(category, args.dataset_type)) 
        args.traindir = "./datasets/ObjectDiscovery-data/Data/%s100"%category
        args.testdir = "./datasets/ObjectDiscovery-data/Data/%s100"%category
        args.savedir = "./data/result/%s-%s100"%(args.dataset_type, category)
        val_image_ids = gen_objdis_imageids(args.traindir)
    else:
        raise Exception("no such dataset type: %s"%args.dataset_type)

    print("images of category: %s: %d"%(category, len(val_image_ids)))
    
    if not osp.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.gpu != "":
        use_cuda=True
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    else:
        use_cuda=False

    if args.algorithm == "ddt":
        ddt = DDT(use_cuda=use_cuda)
        trans_vectors, descriptor_means = ddt.fit(args.traindir, image_ids=val_image_ids)
        ddt.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means, image_ids=val_image_ids)
    elif args.algorithm == "ddtplus":
        ddt_plus = DDTplus(args.select_layers, use_cuda=use_cuda)
        trans_vectors, descriptor_means = ddt_plus.fit(args.traindir, image_ids=val_image_ids)
        ddt_plus.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means, image_ids=val_image_ids)
    else:
        raise Exception("algorithm: %s has not been implemented yet!"%args.algorithm)
        

if __name__ == "__main__":
    main()

