#encoding=utf-8

'''
use densecrf to improve the DDT results
densecrf: https://github.com/lucasb-eyer/pydensecrf
'''

import numpy as np 
import pydensecrf.densecrf as dcrf
import cv2

def do_crf(img, P1, stride=16, niters=5):
    '''
    param img narray: origin rgb image 
    param P1: the projection of the max PCA compolent
    '''
    nlabels = 2
    p_min=np.min(P1)
    normalized = (P1-p_min)/(np.max(P1)-p_min)
    normalized[np.where(normalized>0.5)]=1
    normalized[np.where(normalized<0.1)]=0

    if len(img.shape)==2 or img.shape[-1]!=3:
        raise Exception("gray image not supported")
    
    h, w = img.shape[:2]
    pos_label = cv2.resize(normalized, (w, h)).astype(np.float32)
    neg_label = 1-pos_label
    u=np.stack((pos_label, neg_label))

    u=u.reshape((nlabels,-1))
    
    d = dcrf.DenseCRF2D(w, h, nlabels)  # width, height, nlabels
    d.setUnaryEnergy(u)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
    Q = d.inference(niters)
    map = np.argmax(Q, axis=0).reshape((w,h))
    
    
    ret = np.zeros((w, h))
    ret[np.where(map==0)]=1
    ret = cv2.resize(ret, (P1.shape[0], P1.shape[1]))
    
    return ret

if __name__=="__main__":
    P1=np.random.randn(4,2)
    img = (np.random.randn(64, 32, 3)*255).astype(np.uint8)

    ret = do_crf(img, P1)

    print(ret)

    #print(P1)
