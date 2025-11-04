auto_scale_lr = dict(base_batch_size=16, enable=False)
backend = 'pillow'
backend_args = None
data_root = 'data/LIS/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        test_out_dir=
        '/media/ubuntu/DATA2T/mmdetection-main/work_dirs/condinst_r50_fpn_ms-poly-90k_coco_instance/output',
        type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/media/ubuntu/DATA2T/mmdetection-main/work_dirs/condinst_r50_fpn_ms-poly-90k_coco_instance/iter_70000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
max_iter = 90000
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='ResNet'),
    bbox_head=dict(
        center_sampling=True,
        centerness_on_reg=True,
        conv_bias=True,
        dcn_on_last_conv=False,
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='GIoULoss'),
        loss_centerness=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_on_bbox=True,
        num_classes=80,
        num_params=169,
        stacked_convs=4,
        strides=[
            8,
            16,
            32,
            64,
            128,
        ],
        type='CondInstBboxHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    mask_head=dict(
        feat_channels=8,
        loss_mask=dict(
            activate=True,
            eps=5e-06,
            loss_weight=1.0,
            type='DiceLoss',
            use_sigmoid=True),
        mask_feature_head=dict(
            end_level=2,
            feat_channels=128,
            in_channels=256,
            mask_stride=8,
            norm_cfg=dict(requires_grad=True, type='BN'),
            num_stacked_convs=4,
            out_channels=8,
            start_level=0),
        mask_out_stride=4,
        max_masks_to_train=300,
        num_layers=3,
        size_of_interest=8,
        type='CondInstMaskHead'),
    neck=dict(
        add_extra_convs='on_output',
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=1,
        type='FPN'),
    test_cfg=dict(
        mask_thr=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    type='CondInst')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=0.001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=90000,
        gamma=0.1,
        milestones=[
            60000,
            80000,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/annotations/lis_coco_JPG_test+1.json',
        backend_args=None,
        data_prefix=dict(img='RGB-dark/'),
        data_root='data/LIS/',
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='Resize'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/LIS/annotations/annotations/lis_coco_JPG_test+1.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        backend='pillow', keep_ratio=True, scale=(
            1333,
            800,
        ), type='Resize'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    max_iters=90000, type='IterBasedTrainLoop', val_interval=10000)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/annotations/lis_coco_JPG_train+1.json',
        backend_args=None,
        data_prefix=dict(img='RGB-dark/'),
        data_root='data/LIS/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                backend='pillow',
                keep_ratio=True,
                scales=[
                    (
                        1333,
                        640,
                    ),
                    (
                        1333,
                        672,
                    ),
                    (
                        1333,
                        704,
                    ),
                    (
                        1333,
                        736,
                    ),
                    (
                        1333,
                        768,
                    ),
                    (
                        1333,
                        800,
                    ),
                ],
                type='RandomChoiceResize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(
        backend_args=None, imdecode_backend='pillow',
        type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        backend='pillow',
        keep_ratio=True,
        scales=[
            (
                1333,
                640,
            ),
            (
                1333,
                672,
            ),
            (
                1333,
                704,
            ),
            (
                1333,
                736,
            ),
            (
                1333,
                768,
            ),
            (
                1333,
                800,
            ),
        ],
        type='RandomChoiceResize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/annotations/lis_coco_JPG_test+1.json',
        backend_args=None,
        data_prefix=dict(img='RGB-dark/'),
        data_root='data/LIS/',
        pipeline=[
            dict(
                backend_args=None,
                imdecode_backend='pillow',
                type='LoadImageFromFile'),
            dict(
                backend='pillow',
                keep_ratio=True,
                scale=(
                    1333,
                    800,
                ),
                type='Resize'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/LIS/annotations/annotations/lis_coco_JPG_test+1.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/condinst_r50_fpn_ms-poly-90k_coco_instance'
