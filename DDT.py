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


class DDT(object):
    def __init__(self):
        #加载预训练模型
        self.pretrained_feature_model = models.vgg19(pretrained=True).features

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()

    def fit(self, traindir):
        train_dataset = ImageSet(traindir, resize=1000)

        descriptors = np.zeros((1, 512))

        for index in range(len(train_dataset)):
            print("processing "+str(index)+"th training images.")
            image = train_dataset[index]
            h, w = image.shape[:2]
            image = self.normalize(self.totensor(image)).view(1, 3, h, w)

            output = self.pretrained_feature_model(image)[0, :]
            output = output.view(512, output.shape[1] * output.shape[2])
            output = output.transpose(0, 1)
            descriptors = np.vstack((descriptors, output.detach().numpy().copy()))
            del output

        descriptors = descriptors[1:]

        #计算descriptor均值，并将其降为0
        descriptors_mean = sum(descriptors)/len(descriptors)
        descriptors_mean_tensor = torch.FloatTensor(descriptors_mean)

        pca = PCA(n_components=1)
        pca.fit(descriptors)
        trans_vec = pca.components_[0]
        return trans_vec, descriptors_mean_tensor

    def co_locate(self, testdir, savedir, trans_vector, descriptor_mean_tensor):
        test_dataset = ImageSet(testdir, resize=1000)
        for index in range(len(test_dataset)):
            image = test_dataset[index]
            origin_image = image.copy()
            origin_height, origin_width = origin_image.shape[:2]
            image = self.normalize(self.totensor(image)).view(1, 3, origin_height, origin_width)

            featmap = self.pretrained_feature_model(image)[0, :]
            h, w = featmap.shape[1], featmap.shape[2]
            featmap = featmap.view(512, -1).transpose(0, 1)
            featmap -= descriptor_mean_tensor.repeat(featmap.shape[0], 1)
            features = featmap.detach().numpy()
            del featmap

            P = np.dot(trans_vector, features.transpose()).reshape(h, w)

            mask = self.max_conn_mask(P, origin_height, origin_width)
            mask_3 = np.concatenate(
                (np.zeros((2, origin_height, origin_width), dtype=np.uint16), mask * 255), axis=0)
            #将原图同mask相加并展示
            mask_3 = np.transpose(mask_3, (1, 2, 0))
            mask_3 = origin_image + mask_3
            mask_3[mask_3[:, :, 2] > 254, 2] = 255
            mask_3 = np.array(mask_3, dtype=np.uint8)
            print("save the " + str(index) + "th image. ")
            cv2.imwrite(savedir + str(index) + ".jpg", mask_3)

    def max_conn_mask(self, P, origin_height, origin_width):
        h, w = P.shape[0], P.shape[1]
        highlight = np.zeros(P.shape)
        for i in range(h):
            for j in range(w):
                if P[i][j] > 0:
                    highlight[i][j] = 1

        # 寻找最大的全联通分量
        labels = measure.label(highlight, neighbors=4, background=0)
        props = measure.regionprops(labels)
        max_index = 0
        for i in range(len(props)):
            if props[i].area > props[max_index].area:
                max_index = i
        max_prop = props[max_index]
        highlights_conn = np.zeros(highlight.shape)
        for each in max_prop.coords:
            highlights_conn[each[0]][each[1]] = 1

        # 最近邻插值：
        highlight_big = cv2.resize(highlights_conn,
                                   (origin_width, origin_height),
                                   interpolation=cv2.INTER_NEAREST)

        highlight_big = np.array(highlight_big, dtype=np.uint16).reshape(1, origin_height, origin_width)
        #highlight_3 = np.concatenate((np.zeros((2, origin_height, origin_width), dtype=np.uint16), highlight_big * 255), axis=0)
        return highlight_big



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', type=str, default='./data/train/badminton/', help="the train data folder path.")
    parser.add_argument('--testdir', type=str, default='./data/test/', help="the test data folder path.")
    parser.add_argument('--savedir', type=str, default='./data/result/', help="the final result saving path. ")
    args = parser.parse_args()
    ddt = DDT()
    trans_vectors, descriptor_means = ddt.fit(args.traindir)
    ddt.co_locate(args.testdir, args.savedir, trans_vectors, descriptor_means)
