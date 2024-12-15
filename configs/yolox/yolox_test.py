_base_ = [
    './yolox_s_simota_8xb8-300e_coco.py'
]

img_scale = (640, 640)
backend_args = None
data_root = 'data/coco/'
dataset_type = 'CocoDataset'



train_pipeline = [
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

# Switch to InfiniteSampler to avoid dataloader restart
train_dataloader = dict(sampler=dict(type='DefaultSampler', shuffle=False))
train_dataset = dict(_delete_=True, 
        dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args),
        pipeline=train_pipeline)

randomness = dict(
    seed = 2023,
    diff_rank_seed=True,
    deterministic=True
)