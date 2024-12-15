import torch
import numpy as np
import matplotlib.pyplot as plt
import statistics

def calc_ious(a, b):
    """
    a: [xmin, ymin, xmax, ymax]
    b: [[xmin, ymin, xmax, ymax],
        [xmin, ymin, xmax, ymax],
        ...]
        
    return: array([iou, iou, ...])
    """
    import numpy as np

    b = np.asarray(b)
    a_area = (a[  2] - a[  0]) * (a[  3] - a[  1])
    b_area = (b[:,2] - b[:,0]) * (b[:,3] - b[:,1])
    intersection_xmin = np.maximum(a[0], b[:,0])
    intersection_ymin = np.maximum(a[1], b[:,1])
    intersection_xmax = np.minimum(a[2], b[:,2])
    intersection_ymax = np.minimum(a[3], b[:,3])
    intersection_w = np.maximum(0, intersection_xmax - intersection_xmin)
    intersection_h = np.maximum(0, intersection_ymax - intersection_ymin)
    intersection_area = intersection_w * intersection_h
    union_area = a_area + b_area - intersection_area
    return intersection_area / union_area

def GIou(bboxes, gt):
    num_anchors = bboxes.size()[0]
    num_gt = gt.size()[0]
    gious = torch.zeros(num_anchors, num_gt)
    for i in range(num_anchors):
        for j in range(num_gt):
            C = (max(bboxes[i][3], gt[j][3]) - min(bboxes[i][1], gt[j][1])) * (max(bboxes[i][2], gt[j][2]) - min(bboxes[i][0], gt[j][0]))
            union = max((min(bboxes[i][3], gt[j][3]) - max(bboxes[i][1], gt[j][1])), 0) * max((min(bboxes[i][2], gt[j][2]) - max(bboxes[i][0], gt[j][0])), 0)
            gious[i][j] = (C - union) / C
    return gious
l = torch.tensor([0.8408, 0.8214, 0.7700, 0.7497, 0.6938, 0.6912, 0.6758, 0.6313, 0.6135,0.5886]) * 10
print(l)
print(l.var())