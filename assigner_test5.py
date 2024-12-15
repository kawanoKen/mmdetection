from mmdet.models.detectors.yolox import YOLOX
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.apis import DetInferencer
from mmengine.model.base_model.base_model import BaseModel
from mmengine.runner import load_checkpoint
import torch
import os.path as osp
from mmdet.models.task_modules.assigners import SimOTAAssigner_test, OTAAssigner_test ,OTAAssigner_thr, OTAAssigner_gtiou, OTAAssigner_custom
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

for v in runner.train_dataloader:

    v = DetDataPreprocessor().forward(v)
    inputs = v["inputs"]
    gt_instances = [v["data_samples"][i].gt_instances  for i in range(8)]
    ignored_instances = [v["data_samples"][i].ignored_instances  for i in range(8)]

    yolox.bbox_head.assigner = OTAAssigner_test()
    checkpoint_path = "/data3/yamamura/mmdet_custom/mmdetection-kawano/work_dirs/yolox_s_ota_8xb8-300e_coco/epoch_50.pth"
    load_checkpoint(yolox, checkpoint_path)
    cls_scores, bbox_preds, objectnesses = yolox(inputs)
    ota_results, ota_decoded_bboxes, ota_cls_preds, ota_dynamic_ks, ota_topk_ious, ota_topk_index, ota_valid , ota_cost_matrix= yolox.bbox_head.assigner_test(cls_scores, bbox_preds, objectnesses, gt_instances, 8)

    yolox.bbox_head.assigner = SimOTAAssigner_test()
    checkpoint_path = "/data1/kawano/ota/mmdetection_1/work_dirs/yolox_s_simota_8xb8-300e_coco/epoch_50.pth"
    load_checkpoint(yolox, checkpoint_path)
    cls_scores, bbox_preds, objectnesses = yolox(inputs)
    simota_results, simota_decoded_bboxes, simota_cls_preds, simota_dynamic_ks, simota_topk_ious, simota_topk_index, simota_valid, simota_cost_matrix = yolox.bbox_head.assigner_test(cls_scores, bbox_preds, objectnesses, gt_instances, 8)
    break

img_num = 3
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

anchor_index = np.arange(8400)
img_array = inputs[img_num]/256
img_array = [[[img_array[0][j][k], img_array[1][j][k], img_array[2][j][k]] for k in range(640)] for j in range(640)]



import matplotlib.pyplot as plt
from matplotlib import patches
num_gt = len(gt_instances[img_num])
fig = plt.figure(figsize=(20,10 * num_gt))
fig.suptitle('4th image bboxes plot')
ticks = np.linspace(0, 640, 5)
m = torch.nn.Softmax(dim=1)




#gt plot
for i in range(num_gt):
    k = ota_dynamic_ks[img_num][i]
    l = gt_instances[img_num]["labels"][i]
    ax = fig.add_subplot(num_gt, 2,2*i + 1, title = f"dynamic_k = {k}, gt_num = {i + 1}, label = {l}")
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


    for bb in assigned_bboxes_preds:
        x = bb[0]
        y = bb[1]
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        r = patches.Rectangle( xy=(x,y) , width=w, height=h, fill=False) # 四角形のオブジェクト
        ax.add_patch(r)



    assigned_anchors = (simota_results[img_num] == i + 1).detach().numpy()
    assigned_bboxes_preds = simota_decoded_bboxes[img_num][assigned_anchors].detach().numpy()

    k = simota_dynamic_ks[img_num][i]
    n = assigned_anchors.sum()
    ax = fig.add_subplot(num_gt, 2,2*i + 2, title = f"dynamic_k = {k}, num_assign = {n}")
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


# 4. Axesに図形オブジェクト追加・表示
plt.show()
plt.savefig("ota(L)_vs_simota(R)_50e_4thimg.jpg") 

