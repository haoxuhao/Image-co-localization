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
from ImageSet import ImageSet
import os.path as osp
import os
from DDT import DDT
from tqdm import tqdm


class DDTplus(DDT):
    def __init__(self, selected_layers=(34, 36), use_cuda=True):
        assert(len(selected_layers) == 2)
        self.selected_layers = selected_layers
        if not torch.cuda.is_available():
            self.use_cuda=False
        else:
            self.use_cuda=use_cuda
        print("use_cuda = %s"%str(self.use_cuda))

        #加载预训练模型
        self.pretrained_feature_model = models.vgg19(pretrained=True).features
        if self.use_cuda:
            self.pretrained_feature_model = self.pretrained_feature_model.cuda()

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()

    def fit(self, traindir, image_ids=None):
        train_dataset = ImageSet(traindir, image_ids=image_ids, resize=1000)

        descriptors0 = np.zeros((1, 512))
        descriptors1 = np.zeros((1, 512))
        print("fiting...")
        for index in tqdm(range(len(train_dataset))):
            #print("processing "+str(index)+"th training images.")
            image = train_dataset[index]
            h, w = image.shape[:2]
            image = self.normalize(self.totensor(image)).view(1, 3, h, w)
            if self.use_cuda:
                image = image.cuda()
            for i, layer in enumerate(self.pretrained_feature_model):
                image = layer(image)
                if i in self.selected_layers:
                    output = image[0, :].clone()
                    output = output.view(512, output.shape[1]*output.shape[2])
                    output = output.transpose(0, 1)
                    if i == self.selected_layers[0]:
                        descriptors0 = np.vstack((descriptors0, output.cpu().detach().numpy().copy()))
                    else:
                        descriptors1 = np.vstack((descriptors1, output.cpu().detach().numpy().copy()))
                    del output

        descriptors0 = descriptors0[1:]
        descriptors1 = descriptors1[1:]

        #计算descriptor均值，并将其降为0
        descriptors0_mean = sum(descriptors0)/len(descriptors0)
        descriptors0_mean_tensor = torch.FloatTensor(descriptors0_mean)
        descriptors1_mean = sum(descriptors1)/len(descriptors1)
        descriptors1_mean_tensor = torch.FloatTensor(descriptors1_mean)

        pca0 = PCA(n_components=1)
        pca0.fit(descriptors0)
        trans_vec0 = pca0.components_[0]

        pca1 = PCA(n_components=1)
        pca1.fit(descriptors1)
        trans_vec1 = pca1.components_[0]

        return (trans_vec0, trans_vec1), [descriptors0_mean_tensor, descriptors1_mean_tensor]

    def co_locate(self, testdir, savedir, trans_vectors, descriptor_mean_tensors, image_ids=None):
        test_dataset = ImageSet(testdir, image_ids=image_ids, resize=1000)
        result_file = open(osp.join(savedir, "result.txt"), "w")
        print("colocate...")
        for index in tqdm(range(len(test_dataset))):
            image = test_dataset[index]
            img_id = osp.basename(test_dataset.img_paths[index]).split(".")[0]
            origin_image = image.copy()
            origin_height, origin_width = origin_image.shape[:2]
            image = self.normalize(self.totensor(image)).view(1, 3, origin_height, origin_width)
            if self.use_cuda:
                image = image.cuda()
                descriptor_mean_tensors[0] = descriptor_mean_tensors[0].cuda()
                descriptor_mean_tensors[1] = descriptor_mean_tensors[1].cuda()

            for i, layer in enumerate(self.pretrained_feature_model):
                image = layer(image)
                if i in self.selected_layers:
                    featmap = image[0, :].clone()
                    if i == self.selected_layers[0]:
                        h0, w0 = featmap.shape[1], featmap.shape[2]
                        featmap = featmap.view(512, -1).transpose(0, 1)
                        featmap -= descriptor_mean_tensors[0].repeat(featmap.shape[0], 1)
                        features0 = featmap.cpu().detach().numpy()
                    else:
                        h1, w1 = featmap.shape[1], featmap.shape[2]
                        featmap = featmap.view(512, -1).transpose(0, 1)
                        featmap -= descriptor_mean_tensors[1].repeat(featmap.shape[0], 1)
                        features1 = featmap.cpu().detach().numpy()

                    del featmap

            P0 = np.dot(trans_vectors[0], features0.transpose()).reshape(h0, w0)
            P1 = np.dot(trans_vectors[1], features1.transpose()).reshape(h1, w1)

            mask0 = self.max_conn_mask(P0, origin_height, origin_width)
            mask1 = self.max_conn_mask(P1, origin_height, origin_width)
            mask = mask0+mask1
            mask[mask==1] = 0
            mask[mask==2] = 1

            #get bounding boxes
            bboxes = self.get_bboxes(mask)
            #mask = mask1
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

            # print("save the " + str(index) + "th image. ")
            cv2.imwrite(osp.join(savedir, img_id + ".jpg"), mask_3)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, default='./data/car', help="the train data folder path.")
    parser.add_argument('--testdir', type=str, default='./data/car', help="the test data folder path.")
    parser.add_argument('--savedir', type=str, default='./data/result/car', help="the final result saving path. ")
    parser.add_argument('--select_layers', type=tuple, default=(34, 36), help="two selected layers index. Note that the output channel should be 512. ")
    parser.add_argument("--gpu", type=str, default="0", help="cuda device to run")
    args = parser.parse_args()
    if not osp.exists(args.savedir):
        os.makedirs(args.savedir)
    if args.gpu != "":
        use_cuda=True
    else:
        use_cuda=False

    ddt_plus = DDTplus(args.select_layers, use_cuda=use_cuda)
    trans_vectors, descriptor_means = ddt_plus.fit(args.traindir)
    ddt_plus.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means)
