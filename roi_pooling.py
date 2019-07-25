#encoding=utf-8

'''
roi pooling: a pytorch implementation
'''
import numpy as np
import torch

class RoiHead(torch.nn.Module):
    def __init__(self, pooling_mode="max", output_size=(1, 1), feature_stride=16):
        super(RoiHead, self).__init__()
        if pooling_mode == "max":
            self.pool = torch.nn.AdaptiveMaxPool2d(output_size)
        elif pooling_mode == "avg":
            self.pool = torch.nn.AdaptiveAvgPool2d(output_size)
        else:
            raise TypeError("mode: %s not implemented"%pooling_mode)
        self.feature_stride = int(feature_stride)
        self.output_size = output_size
        
    def forward(self, features, images_proposals):
        '''
        Args:
            features(Torch.Tensor): shape: [b, c, h, w]
            images_proposals(list): list of N images' proposals: [[pps, 4],...,[pps, 4]], proposals format: xywh

        return:
            ret(Torch.Tensor): shape: [N, c, output_size[0], output_size[1]]
        '''
        b, c, h, w = features.shape
        N = 0
        for item in images_proposals:
            N += item.shape[0]

        output = torch.zeros(N, c, self.output_size[1], self.output_size[0])
        k=0
        for i in range(b):
            item = images_proposals[i]
            for j in range(item.shape[0]):
                x, y, w, h = item[j,:]//self.feature_stride
                roi_data = self.pool(features[i, :, y:y+h, x:x+w])
                output[k, :, :, :] = roi_data
                k+=1

        return output

if __name__ == "__main__":
    net = RoiHead()
    input_data = torch.rand(2, 512, 64, 64)
    net = net.cuda()
    input_data=input_data.cuda()
    rois = [np.array([[0,0,40,40],[10,14, 56, 46]]), \
        np.array([[0,0,40,40],[10,14, 56, 46]])]
    out = net(input_data, rois)
    print(out.view(out.shape[1], out.shape[0]*out.shape[2]*out.shape[3]).shape)
    print(out)
    print(out.shape)
        
        
    
