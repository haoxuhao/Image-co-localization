#encoding=utf-8
import argparse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import numpy as np
from numpy import ndarray
from skimage import measure
import cv2
from ImageSet import ImageSet, get_dataloader
import os.path as osp
import os
from utils import gen_voc_imageids, gen_objdis_imageids
from config import Dataset
from tqdm import tqdm
import sys
from models.vgg19_pt_mcn import Vgg19_pt_mcn, vgg19_pt_mcn
from densecrf import do_crf
from roi_pooling import RoiHead
from utils import load_rois
import torch.nn.functional as F



class Pps_DDT(object):
    def __init__(self, use_cuda=False):
        #加载预训练模型
        if not torch.cuda.is_available():
            self.use_cuda=False
        else:
            self.use_cuda=use_cuda
        print("use_cuda = %s"%str(self.use_cuda))
        self.pretrained_feature_model = models.vgg19(pretrained=True).features[:27]
        self.roi_pool = RoiHead()
        #self.pretrained_feature_model.load_state_dict(torch.load("models/vgg19_mcn.pth"))
        if self.use_cuda:
            self.pretrained_feature_model = self.pretrained_feature_model.cuda()
            self.roi_pool = self.roi_pool.cuda()

        self.feature_dim = 512

        # self.normalize = transforms.Normalize(mean=[123.68,  116.779, 103.939],#,
        #                              std=[1,1 ,1])#, [1,1,1]
        
        self.transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

        self.data_loader = None
        self.traindir = None
        self.trainids = None
        self.resize = 800

    def fit(self, traindir, image_ids=None):
        self.data_loader = get_dataloader(traindir, transform = self.transform, image_ids=image_ids, resize=self.resize)
        self.traindir = traindir
        self.trainids = image_ids
        train_loader = self.data_loader
        nsamples = 0
        descriptors = np.zeros((1, self.feature_dim))
        print("fiting...")
        for index, image in tqdm(enumerate(train_loader)):
            if self.use_cuda:
                image=image.cuda()

            output = self.pretrained_feature_model(image)[0, :]
            #output = F.normalize(output, p=2, dim=0)
            #output = output.view(self.feature_dim, output.shape[1] * output.shape[2])
            nsamples+=output.shape[1]
            output = self.pretrained_feature_model(image)
            rois = load_rois(train_loader.dataset.img_paths[index])
            if rois is not None:
                output = self.roi_pool(output, [rois])
            output = output.view(self.feature_dim, output.shape[0]*output.shape[2]*output.shape[3])
            output = output.transpose(0, 1)
            descriptors = np.vstack((descriptors, output.detach().cpu().numpy().copy()))
            del output
        print("nsamples: ", nsamples)
        descriptors = descriptors[1:]

        #计算descriptor均值，并将其降为0
        descriptors_mean = sum(descriptors)/len(descriptors)
        descriptors_mean_tensor = torch.FloatTensor(descriptors_mean)
        pca = PCA()#n_components=1
        pca.fit(descriptors)
        trans_vec = pca.components_[0]
        return trans_vec, descriptors_mean_tensor

    def co_locate(self, testdir,  savedir, trans_vector, descriptor_mean_tensor, image_ids=None):
        is_imageids_same = (set(self.trainids) == set(image_ids)) if (image_ids is not None and self.trainids is not None) else True
        if (testdir == self.traindir) and is_imageids_same:
            test_loader = self.data_loader
        else:
            test_loader = get_dataloader(testdir, transform=self.transform, image_ids=image_ids, resize=self.resize)
        
        if self.use_cuda:
            descriptor_mean_tensor = descriptor_mean_tensor.cuda()
        print("colocate...")
        result_file = open(osp.join(savedir, "result.txt"), "w")
        for index, image in tqdm(enumerate(test_loader)):
            img_id = osp.basename(test_loader.dataset.img_paths[index]).split(".")[0]
            origin_image = cv2.imread(test_loader.dataset.img_paths[index])
            origin_height, origin_width = origin_image.shape[:2]
           
            if self.use_cuda:
                image = image.cuda()
            # featmap = self.pretrained_feature_model(image)[0, :]
            # #featmap = F.normalize(featmap, p=2, dim=0)
            # h, w = featmap.shape[1], featmap.shape[2]
            # featmap = featmap.view(self.feature_dim, -1).transpose(0, 1)

            output = self.pretrained_feature_model(image)
            rois = load_rois(test_loader.dataset.img_paths[index])
            if rois is not None:
                output = self.roi_pool(output, [rois])
            output = output.view(self.feature_dim, output.shape[0]*output.shape[2]*output.shape[3])
            output = output.transpose(0, 1)
            featmap = output.cuda()
            featmap -= descriptor_mean_tensor.repeat(featmap.shape[0], 1)
            features = featmap.detach().cpu().numpy()
            del featmap

            #P = np.dot(trans_vector, features.transpose()).reshape(h, w)
            #P = do_crf(origin_image, P)

            P = np.dot(trans_vector, features.transpose())
            max_score_idx = np.argmax(P)
            bboxes = []
            for i in range(P.shape[0]):
                if P[i]>0 and i==max_score_idx:
                    bbox = rois[i, :]
                    bboxes.append(bbox)
            mask = np.zeros((1, origin_height, origin_width), dtype=np.uint16)
            # sys.exit()
            # mask = self.max_conn_mask(P, origin_height, origin_width)
            # bboxes = self.get_bboxes(mask)

            mask_3 = np.concatenate(
                (np.zeros((2, origin_height, origin_width), dtype=np.uint16), mask * 255), axis=0)
            #将原图同mask相加并展示
            mask_3 = np.transpose(mask_3, (1, 2, 0))
            mask_3 = origin_image + mask_3
            mask_3[mask_3[:, :, 2] > 254, 2] = 255
            mask_3 = np.array(mask_3, dtype=np.uint8)

            #draw bboxes
            if len(bboxes) == 0:
                result_file.write(img_id + "\n")
            for (x, y, w, h) in bboxes:
                cv2.rectangle(mask_3, (x,y), (x+w, y+h), (0, 255, 0), 2) 
                result_file.write(img_id + " {} {} {} {}\n".format(x, y, x+w, y+h))

            cv2.imwrite(osp.join(savedir, img_id + ".jpg"), mask_3)
        result_file.close()

    def max_conn_mask(self, P, origin_height, origin_width):
        h, w = P.shape[0], P.shape[1]
        highlight = np.zeros(P.shape)
        highlight[np.where(P > 0)] = 1
        highlights_conn = np.zeros(highlight.shape)

        if np.sum(highlight) > 1: #no object in this image
            # 寻找最大的全联通分量
            labels = measure.label(highlight, neighbors=4, background=0)
            props = measure.regionprops(labels)
            
            max_index = 0
            for i in range(len(props)):
                if props[i].area > props[max_index].area:
                    max_index = i
            
            max_prop = props[max_index]
            
            for each in max_prop.coords:
                highlights_conn[each[0]][each[1]] = 1

        # 最近邻插值：
        highlight_big = cv2.resize(highlights_conn,
                                   (origin_width, origin_height),
                                   interpolation=cv2.INTER_NEAREST)

        highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(1, origin_height, origin_width)
        #highlight_3 = np.concatenate((np.zeros((2, origin_height, origin_width), dtype=np.uint16), highlight_big * 255), axis=0)
        return highlight_big
    def get_bboxes(self, bin_img):
        img = np.squeeze(bin_img.copy().astype(np.uint8), axis=(0,))
        if int((cv2.__version__).split(".")[0]) >= 4.0:
            contours, hierarchy= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, hierarchy= cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes=[]
        for c in contours:
            # find bounding box coordinates
            # 现计算出一个简单的边界框
            # c = np.squeeze(c, axis=(1,))
            rect = cv2.boundingRect(c)
            bboxes.append(rect)

        return bboxes

# if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--traindir', type=str, default='./data/car', help="the train data folder path.")
    # parser.add_argument('--testdir', type=str, default='./data/car', help="the test data folder path.")
    # parser.add_argument('--savedir', type=str, default='./data/result/car', help="the final result saving path. ")
    # parser.add_argument("--gpu", type=str, default="0", help="cuda device to run")
    # args = parser.parse_args()

    # args.dataset_type = "voc07"
    # category="aeroplane" #"aeroplane"

    # if args.dataset_type == Dataset.voc07:
    #     if not category in Dataset.voc_classes:
    #         raise Exception("no such category: %s in dataset %s"%(category, args.dataset_type)) 
    #     args.traindir = "./datasets/VOC2007/JPEGImages"
    #     args.testdir = "./datasets/VOC2007/JPEGImages"
    #     args.savedir = "./data/result/%s-%s"%(args.dataset_type, category)
    #     val_image_ids = gen_voc_imageids("./datasets/VOC2007", category)
    # elif args.dataset_type == Dataset.objdis:
    #     if not category in Dataset.objdis_classes:
    #         raise Exception("no such category: %s in dataset %s"%(category, args.dataset_type)) 
    #     args.traindir = "./datasets/ObjectDiscovery-data/Data/%s100"%category
    #     args.testdir = "./datasets/ObjectDiscovery-data/Data/%s100"%category
    #     args.savedir = "./data/result/%s-%s100"%(args.dataset_type, category)
    #     val_image_ids = gen_objdis_imageids(args.traindir)
    # else:
    #     raise Exception("no such dataset type: %s"%args.dataset_type)

    # print("images of category: %s: %d"%(category, len(val_image_ids)))
    
    # if not osp.exists(args.savedir):
    #     os.makedirs(args.savedir)
    # if args.gpu != "":
    #     use_cuda=True
    #     os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # else:
    #     use_cuda=False

    # ddt = DDT(use_cuda=use_cuda)
    # trans_vectors, descriptor_means = ddt.fit(args.traindir, image_ids=val_image_ids)
    # ddt.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means, image_ids=val_image_ids)
