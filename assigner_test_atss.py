
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.apis import DetInferencer
from mmengine.model.base_model.base_model import BaseModel
from mmengine.runner import load_checkpoint
import torch
import os.path as osp
from mmdet.models.task_modules.assigners import SimOTAAssigner_test, OTAAssigner_test,  ATSSAssigner_yolox_test, UOTAAssigner
import numpy as np

# img = 'mmdetection/demo/demo.jpg'

config = './configs/yolox/yolox_test.py'
# model = init_detector(config)
# result = inference_detector(model, img)
# print(result)

# print(isinstance(img, (list, tuple)))

cfg = Config.fromfile(config)

cfg.work_dir = osp.join('./work_dirs_test',
                                osp.splitext(osp.basename(config))[0])



runner = RUNNERS.build(cfg)
yolox = runner.model.cpu()
# runner._train_dataloader["dataset"]["pipeline"] = [{'type': 'YOLOXHSVRandomAug'}, 
#                                                    {'type': 'Resize', 'scale': (640, 640), 'keep_ratio': True}, 
#                                                    {'type': 'Pad', 'pad_to_square': True, 'pad_val': {'img': (114.0, 114.0, 114.0)}}, 
#                                                    {'type': 'FilterAnnotations', 'min_gt_bbox_wh': (1, 1), 'keep_empty': False}, 
#                                                    {'type': 'PackDetInputs'}
#                                                    ]
k = 1
t = 0
for v in runner.train_dataloader:
    if t == k:
        v = DetDataPreprocessor().forward(v)
        inputs = v["inputs"]
        gt_instances = [v["data_samples"][i].gt_instances  for i in range(8)]
        ignored_instances = [v["data_samples"][i].ignored_instances  for i in range(8)]

        yolox.bbox_head.assigner = UOTAAssigner()
        checkpoint_path = "/data3/yamamura/mmdet_custom/mmdetection-kawano/work_dirs/yolox_s_ota_8xb8-300e_coco/epoch_300.pth"
        load_checkpoint(yolox, checkpoint_path)
        cls_scores, bbox_preds, objectnesses = yolox(inputs)
        ota_results, ota_decoded_bboxes, ota_cls_preds, ota_dynamic_ks, ota_topk_ious, ota_topk_index, ota_valid, _= yolox.bbox_head.assigner_test(cls_scores, bbox_preds, objectnesses, gt_instances, 8)

        yolox.bbox_head.assigner = OTAAssigner_test()
        #checkpoint_path = "/data1/kawano/ota/mmdetection/work_dirs/yolox_s_atss_8xb8-100e_coco/epoch_40.pth"
        checkpoint_path = "/data1/kawano/ota/mmdetection_1/work_dirs/yolox_s_simota_8xb8-300e_coco/epoch_300.pth"
        load_checkpoint(yolox, checkpoint_path)
        cls_scores, bbox_preds, objectnesses = yolox(inputs)
        atss_results, simota_decoded_bboxes, simota_cls_preds, simota_dynamic_ks, simota_topk_ious, simota_topk_index, simota_valid, _ = yolox.bbox_head.assigner_test(cls_scores, bbox_preds, objectnesses, gt_instances, 8)
        break
    t += 1

img_num = 0
img_num_2 = 0
# #gtlabel 8と10比較
# print("\n")
# print("ota_gt_label8の 0scoreと56score と prior(x,y,X,Y) とtopk_iou")
# assigned_anchors = (ota_results[img_num] == 8).detach().numpy()
# print(ota_cls_preds[1][assigned_anchors][:, 0])
# print(ota_cls_preds[1][assigned_anchors][:, 56])
# print(ota_decoded_bboxes[1][assigned_anchors].detach().numpy())
# # print(ota_topk_ious[1][7])
# print("\n")
# print("ota_gt_label10の 0scoreと56scoreとprior(x,y,X,Y)")
# assigned_anchors = (ota_results[img_num] == 10).detach().numpy()
# print(ota_cls_preds[1][assigned_anchors][:, 0])
# print(ota_cls_preds[1][assigned_anchors][:, 56])
# print(ota_decoded_bboxes[1][assigned_anchors].detach().numpy())
# # print(ota_topk_ious[1][9])

priors = torch.zeros(8400, 4)
for i in range(6400):
    priors[i][0] = (i % 80 )* 8
    priors[i][1] = (i // 80) * 8
    priors[i][2] = 8
    priors[i][3] = 8

for i in range(1600):
    priors[i + 6400][0] = (i % 40) * 16
    priors[i + 6400][1] = (i // 40) * 16
    priors[i + 6400][2] = 16
    priors[i + 6400][3] = 16

for i in range(400):
    priors[i + 8000][0] = (i % 20) * 32
    priors[i + 8000][1] = (i // 20) * 32
    priors[i + 8000][2] = 32
    priors[i + 8000][3] = 32


import matplotlib.pyplot as plt
from matplotlib import patches
num_gt = len(gt_instances[img_num])
fig = plt.figure(figsize=(20,10 * num_gt))
fig.suptitle('7th image bboxes plot')
ticks = np.linspace(0, 640, 5)
m = torch.nn.Softmax(dim=1)

img_array = inputs[img_num]/256
img_array = [[[img_array[0][j][k], img_array[1][j][k], img_array[2][j][k]] for k in range(640)] for j in range(640)]
#gt plot
for i in range(num_gt):
    l = gt_instances[img_num]["labels"][i]
    ax = fig.add_subplot(num_gt, 2,2*i + 1)
    
    ax.imshow(img_array)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid()
    gt_bboxes = gt_instances[img_num]["bboxes"][i]
    x_gt = gt_bboxes[0]
    y_gt = gt_bboxes[1]
    w_gt = gt_bboxes[2] - gt_bboxes[0]
    h_gt = gt_bboxes[3] - gt_bboxes[1]
    gt = patches.Rectangle( xy=(x_gt,y_gt) , width=w_gt, height=h_gt, color=(1, 0, 0), fill=False) # 四角形のオブジェクト
    ax.add_patch(gt)

    #i+1番目labelへの割り当てされたanchor
    assigned_anchors = (ota_results[img_num] == i + 1).detach().numpy()
    assigned_bboxes_preds = ota_decoded_bboxes[img_num][assigned_anchors].detach().numpy()
    assigned_priors = priors[assigned_anchors]


    for bb in assigned_bboxes_preds:
        x = bb[0]
        y = bb[1]
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        r = patches.Rectangle( xy=(x,y) , width=w, height=h, fill=False) # 四角形のオブジェクト
        ax.add_patch(r)
    for p in  assigned_priors:
        r = patches.Rectangle( xy=(p[0],p[1]) , width=p[2], height=p[3], fill=False, color=(0, 1, 0)) # 四角形のオブジェクト
        ax.add_patch(r)
    
    gt_bboxes = gt_instances[img_num_2]["bboxes"][i]
    x_gt = gt_bboxes[0]
    y_gt = gt_bboxes[1]
    w_gt = gt_bboxes[2] - gt_bboxes[0]
    h_gt = gt_bboxes[3] - gt_bboxes[1]
    assigned_anchors = (atss_results[img_num_2] == i + 1).detach().numpy()
    assigned_bboxes_preds = simota_decoded_bboxes[img_num_2][assigned_anchors].detach().numpy()
    assigned_priors = priors[assigned_anchors]
    ax = fig.add_subplot(num_gt, 2,2*i + 2, title = "atss_simota")
    ax.imshow(img_array)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid()
    gt_2 = patches.Rectangle( xy=(x_gt,y_gt) , width=w_gt, height=h_gt, color=(1, 0, 0), fill=False)
    ax.add_patch(gt_2)


    for bb in assigned_bboxes_preds:
        x = bb[0]
        y = bb[1]
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        r = patches.Rectangle( xy=(x,y) , width=w, height=h, fill=False) # 四角形のオブジェクト
        ax.add_patch(r)

    for p in  assigned_priors:
        r = patches.Rectangle( xy=(p[0],p[1]) , width=p[2], height=p[3], fill=False, color=(0, 1, 0)) # 四角形のオブジェクト
        ax.add_patch(r)

# 4. Axesに図形オブジェクト追加・表示
plt.show()
plt.savefig("ota(L)_vs_simota(R)_300e_9thimg_anchorshow.jpg") 

