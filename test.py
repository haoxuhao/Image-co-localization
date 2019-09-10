#!/usr/bin/env python

from DDT import *
from DDTplus import *
from pps_ddt import Pps_DDT
import os.path as osp
import os
from utils import gen_voc_imageids, gen_objdis_imageids
from config import Dataset
import argparse
from cluster.spherical_cluster_method import Sphere

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, default='./data/car', help="the train data folder path.")
    parser.add_argument('--testdir', type=str, default='./data/car', help="the test data folder path.")
    parser.add_argument('--savedir', type=str, default='./data/result/tmp', help="the final result saving path. ")
    parser.add_argument('--select_layers', type=tuple, default=(34, 36), help="two selected layers index. Note that the output channel should be 512. ")
    parser.add_argument("--gpu", type=str, default="0", help="cuda device to run")
    parser.add_argument("--algorithm", "-a", type=str, default="ddt", help="algorithm to use")
    parser.add_argument("--dataset_type", "-dset_type", type=str, default="objdis", help="dataset type")
    parser.add_argument("--category", "-cate", type=str, default=None, help="category")
    args = parser.parse_args()
    return args

def co_locate(model, args):

    if args.dataset_type == Dataset.voc07:
        if not args.category in Dataset.voc_classes:
            raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
        args.traindir = "./datasets/VOC2007/JPEGImages"
        args.testdir = "./datasets/VOC2007/JPEGImages"
        args.savedir = "./data/result/%s-%s"%(args.dataset_type, args.category)
        val_image_ids = gen_voc_imageids("./datasets/VOC2007", args.category)
    elif args.dataset_type == Dataset.voc12:
        if not args.category in Dataset.voc_classes:
            raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
        args.traindir = "./datasets/VOC2012/JPEGImages"
        args.testdir = "./datasets/VOC2012/JPEGImages"
        args.savedir = "./data/result/%s-%s"%(args.dataset_type, args.category)
        val_image_ids = gen_voc_imageids("./datasets/VOC2012", args.category)
    elif args.dataset_type == Dataset.objdis:
        if not args.category in Dataset.objdis_classes:
            raise Exception("no such category: %s in dataset %s"%(args.category, args.dataset_type)) 
        args.traindir = "./datasets/ObjectDiscovery-data/Data/%s100"%args.category
        args.testdir = "./datasets/ObjectDiscovery-data/Data/%s100"%args.category
        args.savedir = "./data/result/%s-%s100"%(args.dataset_type, args.category)
        val_image_ids = gen_objdis_imageids(args.traindir)
    else:
        val_image_ids = [file.split(".")[0] for file in os.listdir(args.traindir)]
        

    print("images of category: %s: %d"%(args.category, len(val_image_ids)))
    if not osp.exists(args.savedir):
        os.makedirs(args.savedir)
    trans_vectors, descriptor_means = model.fit(args.traindir, image_ids=val_image_ids)
    model.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means, image_ids=val_image_ids)
    

def main():
    args = parse_arg()
    #args.dataset_type = "objdis"
    if args.category and args.dataset_type in [Dataset.objdis, Dataset.voc07, Dataset.voc12]:
        args.categories=[args.category]
    elif args.dataset_type == Dataset.voc07:
        args.categories = Dataset.voc_classes
    elif args.dataset_type == Dataset.objdis:
        args.categories = Dataset.objdis_classes
    elif args.dataset_type == Dataset.voc12:
        args.categories = Dataset.voc_classes
    else:
        raise Exception("datset type not found: %s"%args.dataset_type)

    if args.gpu != "":
        use_cuda=True
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    else:
        use_cuda=False
    if args.algorithm == "ddt":
        model = DDT(use_cuda=use_cuda)
    elif args.algorithm == "ddtplus":
        model = DDTplus(args.select_layers, use_cuda=use_cuda)
    elif args.algorithm == "Pps_DDT":
        model = Pps_DDT(use_cuda=use_cuda)
    elif args.algorithm == "sphere_cluster":
        model = Sphere(use_cuda=use_cuda)
    else:
        raise Exception("algorithm: %s has not been implemented yet!"%args.algorithm) 

    for category in tqdm(args.categories):
        args.category = category
        co_locate(model, args)

if __name__ == "__main__":
    main()

