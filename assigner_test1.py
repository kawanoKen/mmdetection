from mmdet.models.detectors.yolox import YOLOX
from mmengine.config import Config
from mmengine.registry import RUNNERS
from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmengine.model.base_model.base_model import BaseModel
import os
import os.path as osp
import torch

from mmdet.models.task_modules.assigners import SimOTAAssigner

# img = 'mmdetection/demo/demo.jpg'

config = './configs/yolox/yolox_test.py'
# model = init_detector(config)
# result = inference_detector(model, img)
# print(result)

# print(isinstance(img, (list, tuple)))

torch.manual_seed(0)
cfg = Config.fromfile(config)

cfg.work_dir = osp.join('./work_dirs_test',
                                osp.splitext(osp.basename(config))[0])


model = cfg.model
backbone = model.backbone
neck = model.neck
head = model.bbox_head
data_preprocessor = DetDataPreprocessor()

runner = RUNNERS.build(cfg)
# runner._train_dataloader["dataset"]["pipeline"] = [{'type': 'YOLOXHSVRandomAug'}, 
#                                                    {'type': 'Resize', 'scale': (640, 640), 'keep_ratio': True}, 
#                                                    {'type': 'Pad', 'pad_to_square': True, 'pad_val': {'img': (114.0, 114.0, 114.0)}}, 
#                                                    {'type': 'FilterAnnotations', 'min_gt_bbox_wh': (1, 1), 'keep_empty': False}, 
#                                                    {'type': 'PackDetInputs'}
#                                                    ]

for v in runner.train_dataloader:

    v = data_preprocessor.forward(v)
    inputs = v["inputs"]
    gt_instances = [v["data_samples"][i].gt_instances  for i in range(8)]
    ignored_instances = [v["data_samples"][i].ignored_instances  for i in range(8)] 
    
    train_cfg = dict(assigner=dict(type='OTAAssigner', center_radius=2.5))
    yolox = YOLOX(backbone, neck, head, train_cfg = train_cfg)

    cls_scores, bbox_preds, objectnesses = yolox._forward(inputs)
    results = yolox.bbox_head.assigner_test(cls_scores, bbox_preds, objectnesses, gt_instances, 8)

    ota_assign_results_stride8 = [results[i][:6400] for i in range (8)]
    ota_assign_results_stride16 = [results[i][6400:8000] for i in range (8)]
    ota_assign_results_stride32 = [results[i][8000:] for i in range (8)]

    train_cfg = dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5))
    yolox = YOLOX(backbone, neck, head, train_cfg = train_cfg)

    cls_scores, bbox_preds, objectnesses = yolox._forward(inputs)
    results = yolox.bbox_head.assigner_test(cls_scores, bbox_preds, objectnesses, gt_instances, 8)

    simota_assign_results_stride8 = [results[i][:6400] for i in range (8)]
    simota_assign_results_stride16 = [results[i][6400:8000] for i in range (8)]
    simota_assign_results_stride32 = [results[i][8000:] for i in range (8)]

    break

breakpoint()

def diff(assigner_1, assigner_2):
    #augs:      two assign results of same stride
    #returns:   values1 & values2(list(list)): differce values from assigner_1 and assigner_2. shape is (num_imgs)(num_priors)
    #           indexes(list(list)): these index. shape is the same

    num_imgs = len(assigner_1)
    num_priors = len(assigner_1[0])
    diff_idx_imgs = []
    diff_values1_imgs = []
    diff_values2_imgs = []
    for i in range(num_imgs):
        diff_idx = []
        diff_values1 = []
        diff_values2 = []

        for j in range(num_priors):
            a = assigner_1[i][j]
            b = assigner_2[i][j]
            if a != b:
                diff_idx.append(j)
                diff_values1.append(a)
                diff_values2.append(b)

        diff_idx_imgs.append(diff_idx)
        diff_values1_imgs.append(diff_values1)
        diff_values2_imgs.append(diff_values2)

    return diff_idx_imgs, diff_values1_imgs, diff_values2_imgs

idxes_8, diffval_ota_8, diffval_simota_8 = diff(ota_assign_results_stride8, simota_assign_results_stride8)
idxes_16, diffval_ota_16, diffval_simota_16 = diff(ota_assign_results_stride16, simota_assign_results_stride16)
idxes_32, diffval_ota_32, diffval_simota_32 = diff(ota_assign_results_stride32, simota_assign_results_stride32)

breakpoint()