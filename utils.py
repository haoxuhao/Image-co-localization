import numpy as np

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

if __name__ == "__main__":
    #test one to many iou
    bb = np.array([20, 30, 50, 50])
    BBGT = np.array([[20, 30, 50, 50], [25, 35, 65, 50], [0, 0, 0, 0]])
    iou, idx = iou_one2many(bb, BBGT)
    print(iou, idx)