import os
import cv2
from Rescale import Rescale
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os.path as osp

class ImageSet(Dataset):
    def __init__(self, folder_path, transform = None, image_ids=None, resize=None):
        if not osp.exists(folder_path):
            raise IOError("image folder not exists: %s"%folder_path)

        if image_ids is not None:
            self.img_paths = [osp.join(folder_path, id + ".jpg") for id in image_ids]
        else:
            self.img_paths = [osp.join(folder_path, file) for file in os.listdir(folder_path)
                   if (file.find(".jpg")!=-1 or file.find(".png") != -1)]

        self.nSamples = len(self.img_paths)
        self.resize = resize
        self.transform = transform
        if resize is not None:
            self.rescaler = Rescale(resize)

    def __getitem__(self, index):
        assert index < self.nSamples, 'index range error' 
        img_path = self.img_paths[index]
        if not osp.exists(img_path):
            raise IOError("no such file: %s"%img_path)
        image = cv2.imread(img_path)
        image = image[:,:,::-1].copy() #BGR to RGB
        if self.resize is not None:
            image = self.rescaler(image)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return self.nSamples

def get_dataloader(imagedir, transform=None, image_ids=None, resize=None, batch_size=1):
    loader = DataLoader(
            ImageSet(imagedir, transform=transform, image_ids=image_ids, resize=resize),
            batch_size=batch_size,
            num_workers=4
    )

    return loader

if __name__ == '__main__':
    img0 = cv2.imread('./data/airplane/0029.jpg')
    img1 = cv2.imread('./data/airplane/0042.jpg')
    img2 = cv2.imread("./datasets/VOC2007/JPEGImages/000033.jpg")
    print(img2.shape)
    rescale = Rescale((224, 224))
    img0, img1 = rescale(img0), rescale(img1)

    print("Over...")
