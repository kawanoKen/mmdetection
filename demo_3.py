from mmengine.dataset.base_dataset import BaseDataset
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.utils import get_loading_pipeline

data_root = 'data/coco/'
ann_file='annotations/instances_8.json'
data_prefix=dict(img='train2017/')
img_scale = (640, 640)
backend_args = None

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    # Resize and Pad are for the last 15 epochs when Mosaic,
    # RandomAffine, and MixUp are closed by YOLOXModeSwitchHook.
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(
        type='Pad',
        pad_to_square=True,
        # If the image is three-channel, the pad value needs
        # to be set separately for each channel.
        pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]


pipelines = [dict(type='LoadImageFromFile', backend_args=backend_args),dict(type='LoadAnnotations', with_bbox=True)]

dataset = CocoDataset(ann_file=ann_file,
                      data_root=data_root,
                      data_prefix=data_prefix,
                      pipeline = pipelines,
                      filter_cfg=dict(filter_empty_gt=False, min_size=32),
                      backend_args=backend_args))
annotation = LoadAnnotations()
annotation.transform(dataset)

breakpoint()
print(data_list)