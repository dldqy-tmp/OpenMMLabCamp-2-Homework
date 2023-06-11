```python
import os
os.chdir('mmdetection')
```


```python
!python tools/train.py data/rtmdet_tiny_drink.py
```

    06/11 16:22:32 - mmengine - [4m[97mINFO[0m - 
    ------------------------------------------------------------
    System environment:
        sys.platform: linux
        Python: 3.8.16 (default, Mar  2 2023, 03:21:46) [GCC 11.2.0]
        CUDA available: True
        numpy_random_seed: 1624391513
        GPU 0,1,2,3,4,5,6,7: Tesla V100-SXM2-32GB
        CUDA_HOME: /usr/local/cuda
        NVCC: Cuda compilation tools, release 11.4, V11.4.152
        GCC: gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-16)
        PyTorch: 1.10.1+cu113
        PyTorch compiling details: PyTorch built with:
      - GCC 7.3
      - C++ Version: 201402
      - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
      - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
      - OpenMP 201511 (a.k.a. OpenMP 4.5)
      - LAPACK is enabled (usually provided by MKL)
      - NNPACK is enabled
      - CPU capability usage: AVX512
      - CUDA Runtime 11.3
      - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
      - CuDNN 8.2
      - Magma 2.5.2
      - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 
    
        TorchVision: 0.11.2+cu113
        OpenCV: 4.7.0
        MMEngine: 0.7.4
    
    Runtime environment:
        cudnn_benchmark: False
        mp_cfg: {'mp_start_method': 'fork', 'opencv_num_threads': 0}
        dist_cfg: {'backend': 'nccl'}
        seed: 1624391513
        Distributed launcher: none
        Distributed training: False
        GPU number: 1
    ------------------------------------------------------------
    
    06/11 16:22:35 - mmengine - [4m[97mINFO[0m - Config:
    dataset_type = 'CocoDataset'
    data_root = '/public3/labmember/zhengdh/openmmlab-true-files/mmdetection/data/drink/Drink_284_Detection_coco/'
    metainfo = dict(
        classes=('cola', 'pepsi', 'fanta', 'sprite', 'spring', 'ice', 'scream',
                 'milk', 'red', 'king'))
    NUM_CLASSES = 10
    load_from = None
    MAX_EPOCHS = 20
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 4
    stage2_num_epochs = 10
    base_lr = 0.004
    VAL_INTERVAL = 5
    default_scope = 'mmdet'
    default_hooks = dict(
        timer=dict(type='IterTimerHook'),
        logger=dict(type='LoggerHook', interval=1),
        param_scheduler=dict(type='ParamSchedulerHook'),
        checkpoint=dict(
            type='CheckpointHook',
            interval=10,
            max_keep_ckpts=2,
            save_best='coco/bbox_mAP'),
        sampler_seed=dict(type='DistSamplerSeedHook'),
        visualization=dict(type='DetVisualizationHook'))
    env_cfg = dict(
        cudnn_benchmark=False,
        mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
        dist_cfg=dict(backend='nccl'))
    vis_backends = [dict(type='LocalVisBackend')]
    visualizer = dict(
        type='DetLocalVisualizer',
        vis_backends=[dict(type='LocalVisBackend')],
        name='visualizer')
    log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
    log_level = 'INFO'
    resume = False
    train_cfg = dict(
        type='EpochBasedTrainLoop',
        max_epochs=20,
        val_interval=5,
        dynamic_intervals=[(10, 1)])
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')
    param_scheduler = [
        dict(
            type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
            end=1000),
        dict(
            type='CosineAnnealingLR',
            eta_min=0.0002,
            begin=150,
            end=300,
            T_max=150,
            by_epoch=True,
            convert_to_iter_based=True)
    ]
    optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
        paramwise_cfg=dict(
            norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
    auto_scale_lr = dict(enable=False, base_batch_size=16)
    backend_args = None
    train_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='CachedMosaic',
            img_scale=(640, 640),
            pad_val=114.0,
            max_cached_images=20,
            random_pop=False),
        dict(
            type='RandomResize',
            scale=(1280, 1280),
            ratio_range=(0.5, 2.0),
            keep_ratio=True),
        dict(type='RandomCrop', crop_size=(640, 640)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        dict(
            type='CachedMixUp',
            img_scale=(640, 640),
            ratio_range=(1.0, 1.0),
            max_cached_images=10,
            random_pop=False,
            pad_val=(114, 114, 114),
            prob=0.5),
        dict(type='PackDetInputs')
    ]
    test_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='Resize', scale=(640, 640), keep_ratio=True),
        dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        dict(
            type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                       'scale_factor'))
    ]
    train_dataloader = dict(
        batch_size=8,
        num_workers=4,
        persistent_workers=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        batch_sampler=None,
        dataset=dict(
            type='CocoDataset',
            data_root=
            '/public3/labmember/zhengdh/openmmlab-true-files/mmdetection/data/drink/Drink_284_Detection_coco/',
            metainfo=dict(
                classes=('cola', 'pepsi', 'fanta', 'sprite', 'spring', 'ice',
                         'scream', 'milk', 'red', 'king')),
            ann_file='train_coco.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='CachedMosaic',
                    img_scale=(640, 640),
                    pad_val=114.0,
                    max_cached_images=20,
                    random_pop=False),
                dict(
                    type='RandomResize',
                    scale=(1280, 1280),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(type='RandomCrop', crop_size=(640, 640)),
                dict(type='YOLOXHSVRandomAug'),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Pad', size=(640, 640),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='CachedMixUp',
                    img_scale=(640, 640),
                    ratio_range=(1.0, 1.0),
                    max_cached_images=10,
                    random_pop=False,
                    pad_val=(114, 114, 114),
                    prob=0.5),
                dict(type='PackDetInputs')
            ],
            backend_args=None),
        pin_memory=True)
    val_dataloader = dict(
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root=
            '/public3/labmember/zhengdh/openmmlab-true-files/mmdetection/data/drink/Drink_284_Detection_coco/',
            metainfo=dict(
                classes=('cola', 'pepsi', 'fanta', 'sprite', 'spring', 'ice',
                         'scream', 'milk', 'red', 'king')),
            ann_file='val_coco.json',
            data_prefix=dict(img='images/'),
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='Resize', scale=(640, 640), keep_ratio=True),
                dict(
                    type='Pad', size=(640, 640),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor'))
            ],
            backend_args=None))
    test_dataloader = dict(
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='CocoDataset',
            data_root=
            '/public3/labmember/zhengdh/openmmlab-true-files/mmdetection/data/drink/Drink_284_Detection_coco/',
            metainfo=dict(
                classes=('cola', 'pepsi', 'fanta', 'sprite', 'spring', 'ice',
                         'scream', 'milk', 'red', 'king')),
            ann_file='val_coco.json',
            data_prefix=dict(img='images/'),
            test_mode=True,
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='Resize', scale=(640, 640), keep_ratio=True),
                dict(
                    type='Pad', size=(640, 640),
                    pad_val=dict(img=(114, 114, 114))),
                dict(
                    type='PackDetInputs',
                    meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                               'scale_factor'))
            ],
            backend_args=None))
    val_evaluator = dict(
        type='CocoMetric',
        ann_file=
        '/public3/labmember/zhengdh/openmmlab-true-files/mmdetection/data/drink/Drink_284_Detection_coco/val_coco.json',
        metric=['bbox'],
        format_only=False,
        backend_args=None,
        proposal_nums=(100, 1, 10))
    test_evaluator = dict(
        type='CocoMetric',
        ann_file=
        '/public3/labmember/zhengdh/openmmlab-true-files/mmdetection/data/drink/Drink_284_Detection_coco/val_coco.json',
        metric=['bbox'],
        format_only=False,
        backend_args=None,
        proposal_nums=(100, 1, 10))
    tta_model = dict(
        type='DetTTAModel',
        tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
    tta_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(
            type='TestTimeAug',
            transforms=[[{
                'type': 'Resize',
                'scale': (640, 640),
                'keep_ratio': True
            }, {
                'type': 'Resize',
                'scale': (320, 320),
                'keep_ratio': True
            }, {
                'type': 'Resize',
                'scale': (960, 960),
                'keep_ratio': True
            }],
                        [{
                            'type': 'RandomFlip',
                            'prob': 1.0
                        }, {
                            'type': 'RandomFlip',
                            'prob': 0.0
                        }],
                        [{
                            'type': 'Pad',
                            'size': (960, 960),
                            'pad_val': {
                                'img': (114, 114, 114)
                            }
                        }],
                        [{
                            'type':
                            'PackDetInputs',
                            'meta_keys':
                            ('img_id', 'img_path', 'ori_shape', 'img_shape',
                             'scale_factor', 'flip', 'flip_direction')
                        }]])
    ]
    model = dict(
        type='RTMDet',
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            bgr_to_rgb=False,
            batch_augments=None),
        backbone=dict(
            type='CSPNeXt',
            arch='P5',
            expand_ratio=0.5,
            deepen_factor=0.167,
            widen_factor=0.375,
            channel_attention=True,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='SiLU', inplace=True),
            init_cfg=dict(
                type='Pretrained',
                prefix='backbone.',
                checkpoint=
                'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
            )),
        neck=dict(
            type='CSPNeXtPAFPN',
            in_channels=[96, 192, 384],
            out_channels=96,
            num_csp_blocks=1,
            expand_ratio=0.5,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='SiLU', inplace=True)),
        bbox_head=dict(
            type='RTMDetSepBNHead',
            num_classes=10,
            in_channels=96,
            stacked_convs=2,
            feat_channels=96,
            anchor_generator=dict(
                type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
            bbox_coder=dict(type='DistancePointBBoxCoder'),
            loss_cls=dict(
                type='QualityFocalLoss',
                use_sigmoid=True,
                beta=2.0,
                loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
            with_objectness=False,
            exp_on_reg=False,
            share_conv=True,
            pred_kernel_size=1,
            norm_cfg=dict(type='SyncBN'),
            act_cfg=dict(type='SiLU', inplace=True)),
        train_cfg=dict(
            assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=30000,
            min_bbox_size=0,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.65),
            max_per_img=300))
    train_pipeline_stage2 = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(
            type='RandomResize',
            scale=(640, 640),
            ratio_range=(0.5, 2.0),
            keep_ratio=True),
        dict(type='RandomCrop', crop_size=(640, 640)),
        dict(type='YOLOXHSVRandomAug'),
        dict(type='RandomFlip', prob=0.5),
        dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
        dict(type='PackDetInputs')
    ]
    custom_hooks = [
        dict(
            type='EMAHook',
            ema_type='ExpMomentumEMA',
            momentum=0.0002,
            update_buffers=True,
            priority=49),
        dict(
            type='PipelineSwitchHook',
            switch_epoch=10,
            switch_pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='RandomResize',
                    scale=(640, 640),
                    ratio_range=(0.5, 2.0),
                    keep_ratio=True),
                dict(type='RandomCrop', crop_size=(640, 640)),
                dict(type='YOLOXHSVRandomAug'),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Pad', size=(640, 640),
                    pad_val=dict(img=(114, 114, 114))),
                dict(type='PackDetInputs')
            ])
    ]
    launcher = 'none'
    work_dir = './work_dirs/rtmdet_tiny_drink'
    
    06/11 16:23:13 - mmengine - [4m[97mINFO[0m - Distributed training is not used, all SyncBatchNorm (SyncBN) layers in the model will be automatically reverted to BatchNormXd layers if they are used.
    06/11 16:23:13 - mmengine - [4m[97mINFO[0m - Hooks will be executed in the following order:
    before_run:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (49          ) EMAHook                            
    (BELOW_NORMAL) LoggerHook                         
     -------------------- 
    after_load_checkpoint:
    (49          ) EMAHook                            
     -------------------- 
    before_train:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (49          ) EMAHook                            
    (NORMAL      ) IterTimerHook                      
    (VERY_LOW    ) CheckpointHook                     
     -------------------- 
    before_train_epoch:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (NORMAL      ) IterTimerHook                      
    (NORMAL      ) DistSamplerSeedHook                
    (NORMAL      ) PipelineSwitchHook                 
     -------------------- 
    before_train_iter:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (NORMAL      ) IterTimerHook                      
     -------------------- 
    after_train_iter:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (49          ) EMAHook                            
    (NORMAL      ) IterTimerHook                      
    (BELOW_NORMAL) LoggerHook                         
    (LOW         ) ParamSchedulerHook                 
    (VERY_LOW    ) CheckpointHook                     
     -------------------- 
    after_train_epoch:
    (NORMAL      ) IterTimerHook                      
    (LOW         ) ParamSchedulerHook                 
    (VERY_LOW    ) CheckpointHook                     
     -------------------- 
    before_val_epoch:
    (49          ) EMAHook                            
    (NORMAL      ) IterTimerHook                      
     -------------------- 
    before_val_iter:
    (NORMAL      ) IterTimerHook                      
     -------------------- 
    after_val_iter:
    (NORMAL      ) IterTimerHook                      
    (NORMAL      ) DetVisualizationHook               
    (BELOW_NORMAL) LoggerHook                         
     -------------------- 
    after_val_epoch:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (49          ) EMAHook                            
    (NORMAL      ) IterTimerHook                      
    (BELOW_NORMAL) LoggerHook                         
    (LOW         ) ParamSchedulerHook                 
    (VERY_LOW    ) CheckpointHook                     
     -------------------- 
    before_save_checkpoint:
    (49          ) EMAHook                            
     -------------------- 
    after_train:
    (VERY_LOW    ) CheckpointHook                     
     -------------------- 
    before_test_epoch:
    (49          ) EMAHook                            
    (NORMAL      ) IterTimerHook                      
     -------------------- 
    before_test_iter:
    (NORMAL      ) IterTimerHook                      
     -------------------- 
    after_test_iter:
    (NORMAL      ) IterTimerHook                      
    (NORMAL      ) DetVisualizationHook               
    (BELOW_NORMAL) LoggerHook                         
     -------------------- 
    after_test_epoch:
    (VERY_HIGH   ) RuntimeInfoHook                    
    (49          ) EMAHook                            
    (NORMAL      ) IterTimerHook                      
    (BELOW_NORMAL) LoggerHook                         
     -------------------- 
    after_run:
    (BELOW_NORMAL) LoggerHook                         
     -------------------- 
    loading annotations into memory...
    Done (t=0.08s)
    creating index...
    index created!
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stem.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stem.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stem.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stem.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stem.2.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stem.2.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage1.1.attention.fc.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage2.1.attention.fc.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage3.1.attention.fc.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.1.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.1.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.1.conv2.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.1.conv2.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- backbone.stage4.2.attention.fc.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.reduce_layers.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.reduce_layers.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.reduce_layers.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.reduce_layers.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.0.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.top_down_blocks.1.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.downsamples.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.downsamples.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.downsamples.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.downsamples.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.0.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.main_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.main_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.short_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.short_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.final_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.final_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.blocks.0.conv1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.blocks.0.conv1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.blocks.0.conv2.depthwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.blocks.0.conv2.depthwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.blocks.0.conv2.pointwise_conv.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.bottom_up_blocks.1.blocks.0.conv2.pointwise_conv.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.out_convs.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.out_convs.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.out_convs.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.out_convs.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.out_convs.2.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- neck.out_convs.2.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.0.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.0.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.0.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.0.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.cls_convs.1.0.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.1.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.1.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.cls_convs.1.1.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.1.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.1.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.cls_convs.2.0.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.2.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.2.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.cls_convs.2.1.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.2.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.cls_convs.2.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.0.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.0.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.0.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.0.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.reg_convs.1.0.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.1.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.1.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.reg_convs.1.1.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.1.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.1.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.reg_convs.2.0.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.2.0.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.2.0.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [5m[4m[33mWARNING[0m - bbox_head.reg_convs.2.1.conv is duplicate. It is skipped since bypass_duplicate=True
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.2.1.bn.weight:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.reg_convs.2.1.bn.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.rtm_cls.0.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.rtm_cls.1.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.rtm_cls.2.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.rtm_reg.0.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.rtm_reg.1.bias:weight_decay=0.0
    06/11 16:23:15 - mmengine - [4m[97mINFO[0m - paramwise_options -- bbox_head.rtm_reg.2.bias:weight_decay=0.0
    loading annotations into memory...
    Done (t=0.01s)
    creating index...
    index created!
    loading annotations into memory...
    Done (t=0.00s)
    creating index...
    index created!
    06/11 16:23:18 - mmengine - [4m[97mINFO[0m - load backbone. in model from: https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth
    Loads checkpoint by http backend from path: https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth
    06/11 16:23:19 - mmengine - [5m[4m[33mWARNING[0m - "FileClient" will be deprecated in future. Please use io functions in https://mmengine.readthedocs.io/en/latest/api/fileio.html#file-io
    06/11 16:23:19 - mmengine - [5m[4m[33mWARNING[0m - "HardDiskBackend" is the alias of "LocalBackend" and the former will be deprecated in future.
    06/11 16:23:19 - mmengine - [4m[97mINFO[0m - Checkpoints will be saved to /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink.
    /public3/labmember/zhengdh/miniconda3/envs/openmmlab/lib/python3.8/site-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    06/11 16:23:23 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 1/29]  lr: 4.0000e-08  eta: 0:48:05  time: 4.9842  data_time: 4.4371  memory: 2547  loss: 0.8467  loss_cls: 0.6800  loss_bbox: 0.1666
    06/11 16:23:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 2/29]  lr: 4.0440e-06  eta: 0:25:19  time: 2.6293  data_time: 2.2502  memory: 2611  loss: 0.9528  loss_cls: 0.6800  loss_bbox: 0.2728
    06/11 16:23:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 3/29]  lr: 8.0479e-06  eta: 0:17:43  time: 1.8437  data_time: 1.5341  memory: 2624  loss: 0.9337  loss_cls: 0.6800  loss_bbox: 0.2537
    06/11 16:23:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 4/29]  lr: 1.2052e-05  eta: 0:13:48  time: 1.4383  data_time: 1.1632  memory: 2602  loss: 0.9034  loss_cls: 0.6797  loss_bbox: 0.2237
    06/11 16:23:28 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 5/29]  lr: 1.6056e-05  eta: 0:18:59  time: 1.9817  data_time: 1.7052  memory: 2637  loss: 0.9091  loss_cls: 0.6793  loss_bbox: 0.2298
    06/11 16:23:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 6/29]  lr: 2.0060e-05  eta: 0:16:21  time: 1.7092  data_time: 1.4412  memory: 2619  loss: 1.0096  loss_cls: 0.6790  loss_bbox: 0.3307
    06/11 16:23:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 7/29]  lr: 2.4064e-05  eta: 0:14:16  time: 1.4940  data_time: 1.2408  memory: 2609  loss: 0.9908  loss_cls: 0.6785  loss_bbox: 0.3124
    06/11 16:23:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 8/29]  lr: 2.8068e-05  eta: 0:12:41  time: 1.3320  data_time: 1.0897  memory: 2605  loss: 0.9881  loss_cls: 0.6779  loss_bbox: 0.3101
    06/11 16:23:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][ 9/29]  lr: 3.2072e-05  eta: 0:14:58  time: 1.5733  data_time: 1.3292  memory: 2607  loss: 1.1369  loss_cls: 0.6868  loss_bbox: 0.4501
    06/11 16:23:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][10/29]  lr: 3.6076e-05  eta: 0:13:39  time: 1.4373  data_time: 1.1995  memory: 2624  loss: 1.1394  loss_cls: 0.6852  loss_bbox: 0.4541
    06/11 16:23:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][11/29]  lr: 4.0080e-05  eta: 0:12:35  time: 1.3270  data_time: 1.0963  memory: 2603  loss: 1.1255  loss_cls: 0.6838  loss_bbox: 0.4417
    06/11 16:23:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][12/29]  lr: 4.4084e-05  eta: 0:11:39  time: 1.2324  data_time: 1.0070  memory: 2624  loss: 1.1102  loss_cls: 0.6825  loss_bbox: 0.4278
    06/11 16:23:37 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][13/29]  lr: 4.8088e-05  eta: 0:13:34  time: 1.4358  data_time: 1.2056  memory: 2610  loss: 1.1361  loss_cls: 0.6812  loss_bbox: 0.4549
    06/11 16:23:37 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][14/29]  lr: 5.2092e-05  eta: 0:12:44  time: 1.3500  data_time: 1.1224  memory: 2658  loss: 1.2106  loss_cls: 0.6806  loss_bbox: 0.5299
    06/11 16:23:38 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][15/29]  lr: 5.6095e-05  eta: 0:12:00  time: 1.2753  data_time: 1.0517  memory: 2602  loss: 1.2200  loss_cls: 0.6795  loss_bbox: 0.5405
    06/11 16:23:38 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][16/29]  lr: 6.0099e-05  eta: 0:11:23  time: 1.2123  data_time: 0.9909  memory: 2616  loss: 1.2269  loss_cls: 0.6783  loss_bbox: 0.5486
    06/11 16:23:42 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][17/29]  lr: 6.4103e-05  eta: 0:12:52  time: 1.3721  data_time: 1.1468  memory: 2611  loss: 1.2288  loss_cls: 0.6770  loss_bbox: 0.5519
    06/11 16:23:42 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][18/29]  lr: 6.8107e-05  eta: 0:12:17  time: 1.3120  data_time: 1.0875  memory: 2602  loss: 1.2607  loss_cls: 0.6763  loss_bbox: 0.5845
    06/11 16:23:42 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][19/29]  lr: 7.2111e-05  eta: 0:11:45  time: 1.2578  data_time: 1.0352  memory: 2604  loss: 1.2992  loss_cls: 0.6764  loss_bbox: 0.6228
    06/11 16:23:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][20/29]  lr: 7.6115e-05  eta: 0:11:16  time: 1.2085  data_time: 0.9874  memory: 2632  loss: 1.3405  loss_cls: 0.6776  loss_bbox: 0.6629
    06/11 16:23:46 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][21/29]  lr: 8.0119e-05  eta: 0:12:16  time: 1.3173  data_time: 1.0952  memory: 2604  loss: 1.3615  loss_cls: 0.6769  loss_bbox: 0.6847
    06/11 16:23:46 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][22/29]  lr: 8.4123e-05  eta: 0:11:46  time: 1.2663  data_time: 1.0467  memory: 2608  loss: 1.3928  loss_cls: 0.6757  loss_bbox: 0.7171
    06/11 16:23:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][23/29]  lr: 8.8127e-05  eta: 0:11:22  time: 1.2252  data_time: 1.0065  memory: 2649  loss: 1.4236  loss_cls: 0.6582  loss_bbox: 0.7654
    06/11 16:23:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][24/29]  lr: 9.2131e-05  eta: 0:10:59  time: 1.1865  data_time: 0.9695  memory: 2633  loss: 1.4504  loss_cls: 0.6440  loss_bbox: 0.8065
    06/11 16:23:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][25/29]  lr: 9.6135e-05  eta: 0:12:07  time: 1.3101  data_time: 1.0906  memory: 2605  loss: 1.4775  loss_cls: 0.6346  loss_bbox: 0.8428
    06/11 16:23:52 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][26/29]  lr: 1.0014e-04  eta: 0:11:44  time: 1.2709  data_time: 1.0497  memory: 2604  loss: 1.5046  loss_cls: 0.6302  loss_bbox: 0.8744
    06/11 16:23:52 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][27/29]  lr: 1.0414e-04  eta: 0:11:21  time: 1.2331  data_time: 1.0128  memory: 2628  loss: 1.5262  loss_cls: 0.6211  loss_bbox: 0.9051
    06/11 16:23:52 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][28/29]  lr: 1.0815e-04  eta: 0:11:01  time: 1.1981  data_time: 0.9794  memory: 2602  loss: 1.5565  loss_cls: 0.6188  loss_bbox: 0.9377
    06/11 16:23:53 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:23:53 - mmengine - [4m[97mINFO[0m - Epoch(train)  [1][29/29]  lr: 1.1215e-04  eta: 0:10:53  time: 1.1863  data_time: 0.9699  memory: 1379  loss: 1.5742  loss_cls: 0.6104  loss_bbox: 0.9638
    06/11 16:23:58 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 1/29]  lr: 1.1615e-04  eta: 0:11:55  time: 1.3002  data_time: 1.0814  memory: 2610  loss: 1.6001  loss_cls: 0.6193  loss_bbox: 0.9808
    06/11 16:23:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 2/29]  lr: 1.2016e-04  eta: 0:11:51  time: 1.2952  data_time: 1.0761  memory: 2627  loss: 1.6205  loss_cls: 0.6256  loss_bbox: 0.9949
    06/11 16:24:00 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 3/29]  lr: 1.2416e-04  eta: 0:11:52  time: 1.3010  data_time: 1.0808  memory: 2624  loss: 1.6393  loss_cls: 0.6371  loss_bbox: 1.0022
    06/11 16:24:01 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 4/29]  lr: 1.2817e-04  eta: 0:11:51  time: 1.3012  data_time: 1.0790  memory: 2627  loss: 1.6553  loss_cls: 0.6471  loss_bbox: 1.0081
    06/11 16:24:03 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 5/29]  lr: 1.3217e-04  eta: 0:11:53  time: 1.3061  data_time: 1.0846  memory: 2610  loss: 1.6677  loss_cls: 0.6477  loss_bbox: 1.0200
    06/11 16:24:03 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 6/29]  lr: 1.3617e-04  eta: 0:11:36  time: 1.2771  data_time: 1.0562  memory: 2626  loss: 1.6809  loss_cls: 0.6569  loss_bbox: 1.0240
    06/11 16:24:03 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 7/29]  lr: 1.4018e-04  eta: 0:11:18  time: 1.2478  data_time: 1.0282  memory: 2609  loss: 1.6954  loss_cls: 0.6653  loss_bbox: 1.0301
    06/11 16:24:04 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 8/29]  lr: 1.4418e-04  eta: 0:11:03  time: 1.2214  data_time: 1.0009  memory: 2635  loss: 1.7054  loss_cls: 0.6650  loss_bbox: 1.0404
    06/11 16:24:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][ 9/29]  lr: 1.4819e-04  eta: 0:11:31  time: 1.2753  data_time: 1.0544  memory: 2602  loss: 1.7218  loss_cls: 0.6774  loss_bbox: 1.0445
    06/11 16:24:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][10/29]  lr: 1.5219e-04  eta: 0:11:15  time: 1.2489  data_time: 1.0291  memory: 2611  loss: 1.7318  loss_cls: 0.6820  loss_bbox: 1.0499
    06/11 16:24:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][11/29]  lr: 1.5619e-04  eta: 0:11:00  time: 1.2230  data_time: 1.0046  memory: 2602  loss: 1.7461  loss_cls: 0.6964  loss_bbox: 1.0497
    06/11 16:24:08 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][12/29]  lr: 1.6020e-04  eta: 0:10:47  time: 1.2015  data_time: 0.9840  memory: 2612  loss: 1.7554  loss_cls: 0.7029  loss_bbox: 1.0525
    06/11 16:24:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][13/29]  lr: 1.6420e-04  eta: 0:11:18  time: 1.2620  data_time: 1.0432  memory: 2607  loss: 1.7633  loss_cls: 0.7077  loss_bbox: 1.0556
    06/11 16:24:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][14/29]  lr: 1.6821e-04  eta: 0:11:06  time: 1.2409  data_time: 1.0206  memory: 2624  loss: 1.7706  loss_cls: 0.7097  loss_bbox: 1.0609
    06/11 16:24:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][15/29]  lr: 1.7221e-04  eta: 0:10:54  time: 1.2217  data_time: 0.9997  memory: 2619  loss: 1.7791  loss_cls: 0.7144  loss_bbox: 1.0648
    06/11 16:24:13 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][16/29]  lr: 1.7621e-04  eta: 0:10:42  time: 1.2011  data_time: 0.9797  memory: 2617  loss: 1.7853  loss_cls: 0.7180  loss_bbox: 1.0672
    06/11 16:24:16 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][17/29]  lr: 1.8022e-04  eta: 0:11:04  time: 1.2441  data_time: 1.0220  memory: 2640  loss: 1.7914  loss_cls: 0.7182  loss_bbox: 1.0732
    06/11 16:24:16 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][18/29]  lr: 1.8422e-04  eta: 0:10:52  time: 1.2235  data_time: 1.0016  memory: 2620  loss: 1.7991  loss_cls: 0.7245  loss_bbox: 1.0746
    06/11 16:24:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][19/29]  lr: 1.8823e-04  eta: 0:10:42  time: 1.2082  data_time: 0.9858  memory: 2602  loss: 1.8036  loss_cls: 0.7251  loss_bbox: 1.0785
    06/11 16:24:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][20/29]  lr: 1.9223e-04  eta: 0:10:31  time: 1.1897  data_time: 0.9682  memory: 2615  loss: 1.8090  loss_cls: 0.7292  loss_bbox: 1.0797
    06/11 16:24:19 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][21/29]  lr: 1.9623e-04  eta: 0:10:42  time: 1.2121  data_time: 0.9912  memory: 2605  loss: 1.8123  loss_cls: 0.7282  loss_bbox: 1.0842
    06/11 16:24:19 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][22/29]  lr: 2.0024e-04  eta: 0:10:30  time: 1.1166  data_time: 0.9029  memory: 2619  loss: 1.8331  loss_cls: 0.7289  loss_bbox: 1.1042
    06/11 16:24:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][23/29]  lr: 2.0424e-04  eta: 0:10:29  time: 1.1339  data_time: 0.9205  memory: 2624  loss: 1.8513  loss_cls: 0.7280  loss_bbox: 1.1232
    06/11 16:24:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][24/29]  lr: 2.0825e-04  eta: 0:10:18  time: 1.1342  data_time: 0.9204  memory: 2616  loss: 1.8730  loss_cls: 0.7278  loss_bbox: 1.1452
    06/11 16:24:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][25/29]  lr: 2.1225e-04  eta: 0:10:32  time: 1.1846  data_time: 0.9687  memory: 2614  loss: 1.8986  loss_cls: 0.7329  loss_bbox: 1.1656
    06/11 16:24:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][26/29]  lr: 2.1625e-04  eta: 0:10:23  time: 1.1082  data_time: 0.8922  memory: 2611  loss: 1.9189  loss_cls: 0.7338  loss_bbox: 1.1852
    06/11 16:24:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][27/29]  lr: 2.2026e-04  eta: 0:10:14  time: 1.1091  data_time: 0.8937  memory: 2614  loss: 1.9304  loss_cls: 0.7380  loss_bbox: 1.1924
    06/11 16:24:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][28/29]  lr: 2.2426e-04  eta: 0:10:04  time: 1.1090  data_time: 0.8936  memory: 2614  loss: 1.9500  loss_cls: 0.7394  loss_bbox: 1.2106
    06/11 16:24:25 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:24:25 - mmengine - [4m[97mINFO[0m - Epoch(train)  [2][29/29]  lr: 2.2827e-04  eta: 0:09:59  time: 1.1194  data_time: 0.9043  memory: 1358  loss: 1.9723  loss_cls: 0.7460  loss_bbox: 1.2263
    06/11 16:24:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 1/29]  lr: 2.3227e-04  eta: 0:10:24  time: 1.1310  data_time: 0.9156  memory: 2609  loss: 1.9651  loss_cls: 0.7484  loss_bbox: 1.2166
    06/11 16:24:30 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 2/29]  lr: 2.3627e-04  eta: 0:10:15  time: 1.1318  data_time: 0.9160  memory: 2611  loss: 1.9813  loss_cls: 0.7505  loss_bbox: 1.2308
    06/11 16:24:30 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 3/29]  lr: 2.4028e-04  eta: 0:10:06  time: 1.1328  data_time: 0.9164  memory: 2623  loss: 2.0002  loss_cls: 0.7548  loss_bbox: 1.2454
    06/11 16:24:30 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 4/29]  lr: 2.4428e-04  eta: 0:09:57  time: 1.1342  data_time: 0.9171  memory: 2631  loss: 2.0221  loss_cls: 0.7604  loss_bbox: 1.2617
    06/11 16:24:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 5/29]  lr: 2.4829e-04  eta: 0:10:11  time: 1.1178  data_time: 0.9020  memory: 2606  loss: 2.0311  loss_cls: 0.7632  loss_bbox: 1.2679
    06/11 16:24:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 6/29]  lr: 2.5229e-04  eta: 0:10:03  time: 1.1190  data_time: 0.9033  memory: 2649  loss: 2.0279  loss_cls: 0.7660  loss_bbox: 1.2618
    06/11 16:24:34 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 7/29]  lr: 2.5629e-04  eta: 0:09:54  time: 1.1187  data_time: 0.9029  memory: 2633  loss: 2.0385  loss_cls: 0.7674  loss_bbox: 1.2712
    06/11 16:24:34 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 8/29]  lr: 2.6030e-04  eta: 0:09:48  time: 1.1237  data_time: 0.9076  memory: 2628  loss: 2.0521  loss_cls: 0.7736  loss_bbox: 1.2785
    06/11 16:24:37 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][ 9/29]  lr: 2.6430e-04  eta: 0:10:01  time: 1.1036  data_time: 0.8898  memory: 2618  loss: 2.0646  loss_cls: 0.7770  loss_bbox: 1.2876
    06/11 16:24:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][10/29]  lr: 2.6831e-04  eta: 0:10:02  time: 1.1293  data_time: 0.9157  memory: 2619  loss: 2.0665  loss_cls: 0.7787  loss_bbox: 1.2878
    06/11 16:24:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][11/29]  lr: 2.7231e-04  eta: 0:09:54  time: 1.1284  data_time: 0.9150  memory: 2612  loss: 2.0645  loss_cls: 0.7806  loss_bbox: 1.2839
    06/11 16:24:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][12/29]  lr: 2.7631e-04  eta: 0:09:46  time: 1.1267  data_time: 0.9138  memory: 2615  loss: 2.0586  loss_cls: 0.7817  loss_bbox: 1.2769
    06/11 16:24:41 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][13/29]  lr: 2.8032e-04  eta: 0:09:50  time: 1.0955  data_time: 0.8838  memory: 2616  loss: 2.0600  loss_cls: 0.7847  loss_bbox: 1.2752
    06/11 16:24:42 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][14/29]  lr: 2.8432e-04  eta: 0:09:50  time: 1.1175  data_time: 0.9054  memory: 2608  loss: 2.0553  loss_cls: 0.7865  loss_bbox: 1.2688
    06/11 16:24:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][15/29]  lr: 2.8833e-04  eta: 0:09:42  time: 1.1146  data_time: 0.9034  memory: 2607  loss: 2.0494  loss_cls: 0.7969  loss_bbox: 1.2525
    06/11 16:24:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][16/29]  lr: 2.9233e-04  eta: 0:09:35  time: 1.1125  data_time: 0.9016  memory: 2615  loss: 2.0439  loss_cls: 0.8039  loss_bbox: 1.2400
    06/11 16:24:44 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][17/29]  lr: 2.9633e-04  eta: 0:09:34  time: 1.0510  data_time: 0.8417  memory: 2619  loss: 2.0393  loss_cls: 0.8134  loss_bbox: 1.2260
    06/11 16:24:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][18/29]  lr: 3.0034e-04  eta: 0:09:42  time: 1.0968  data_time: 0.8884  memory: 2630  loss: 2.0299  loss_cls: 0.8164  loss_bbox: 1.2135
    06/11 16:24:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][19/29]  lr: 3.0434e-04  eta: 0:09:35  time: 1.0966  data_time: 0.8885  memory: 2628  loss: 2.0264  loss_cls: 0.8271  loss_bbox: 1.1993
    06/11 16:24:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][20/29]  lr: 3.0835e-04  eta: 0:09:28  time: 1.0959  data_time: 0.8878  memory: 2630  loss: 2.0187  loss_cls: 0.8351  loss_bbox: 1.1836
    06/11 16:24:48 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][21/29]  lr: 3.1235e-04  eta: 0:09:24  time: 1.0935  data_time: 0.8850  memory: 2610  loss: 2.0154  loss_cls: 0.8473  loss_bbox: 1.1681
    06/11 16:24:50 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][22/29]  lr: 3.1635e-04  eta: 0:09:33  time: 1.0565  data_time: 0.8499  memory: 2605  loss: 2.0075  loss_cls: 0.8484  loss_bbox: 1.1592
    06/11 16:24:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][23/29]  lr: 3.2036e-04  eta: 0:09:26  time: 1.0371  data_time: 0.8319  memory: 2623  loss: 2.0024  loss_cls: 0.8523  loss_bbox: 1.1501
    06/11 16:24:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][24/29]  lr: 3.2436e-04  eta: 0:09:19  time: 1.0114  data_time: 0.8081  memory: 2620  loss: 1.9931  loss_cls: 0.8481  loss_bbox: 1.1451
    06/11 16:24:52 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][25/29]  lr: 3.2837e-04  eta: 0:09:16  time: 1.0005  data_time: 0.7994  memory: 2640  loss: 1.9854  loss_cls: 0.8458  loss_bbox: 1.1396
    06/11 16:24:54 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][26/29]  lr: 3.3237e-04  eta: 0:09:26  time: 1.0293  data_time: 0.8287  memory: 2615  loss: 1.9819  loss_cls: 0.8520  loss_bbox: 1.1299
    06/11 16:24:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][27/29]  lr: 3.3637e-04  eta: 0:09:19  time: 1.0281  data_time: 0.8282  memory: 2611  loss: 1.9762  loss_cls: 0.8491  loss_bbox: 1.1272
    06/11 16:24:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][28/29]  lr: 3.4038e-04  eta: 0:09:13  time: 1.0278  data_time: 0.8281  memory: 2645  loss: 1.9699  loss_cls: 0.8487  loss_bbox: 1.1213
    06/11 16:24:55 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:24:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [3][29/29]  lr: 3.4438e-04  eta: 0:09:06  time: 1.0253  data_time: 0.8283  memory: 1374  loss: 1.9649  loss_cls: 0.8527  loss_bbox: 1.1122
    06/11 16:24:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 1/29]  lr: 3.4838e-04  eta: 0:09:23  time: 1.0473  data_time: 0.8495  memory: 2627  loss: 1.9540  loss_cls: 0.8472  loss_bbox: 1.1068
    06/11 16:25:00 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 2/29]  lr: 3.5239e-04  eta: 0:09:17  time: 1.0469  data_time: 0.8491  memory: 2606  loss: 1.9500  loss_cls: 0.8503  loss_bbox: 1.0997
    06/11 16:25:00 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 3/29]  lr: 3.5639e-04  eta: 0:09:11  time: 1.0471  data_time: 0.8491  memory: 2616  loss: 1.9435  loss_cls: 0.8463  loss_bbox: 1.0971
    06/11 16:25:00 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 4/29]  lr: 3.6040e-04  eta: 0:09:05  time: 1.0461  data_time: 0.8482  memory: 2604  loss: 1.9370  loss_cls: 0.8455  loss_bbox: 1.0915
    06/11 16:25:03 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 5/29]  lr: 3.6440e-04  eta: 0:09:15  time: 1.0340  data_time: 0.8374  memory: 2616  loss: 1.9326  loss_cls: 0.8475  loss_bbox: 1.0851
    06/11 16:25:04 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 6/29]  lr: 3.6840e-04  eta: 0:09:11  time: 1.0382  data_time: 0.8433  memory: 2602  loss: 1.9279  loss_cls: 0.8505  loss_bbox: 1.0774
    06/11 16:25:04 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 7/29]  lr: 3.7241e-04  eta: 0:09:05  time: 1.0348  data_time: 0.8423  memory: 2611  loss: 1.9224  loss_cls: 0.8509  loss_bbox: 1.0714
    06/11 16:25:05 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 8/29]  lr: 3.7641e-04  eta: 0:09:02  time: 1.0452  data_time: 0.8514  memory: 2630  loss: 1.9161  loss_cls: 0.8507  loss_bbox: 1.0654
    06/11 16:25:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][ 9/29]  lr: 3.8042e-04  eta: 0:09:06  time: 1.0217  data_time: 0.8281  memory: 2607  loss: 1.9104  loss_cls: 0.8542  loss_bbox: 1.0562
    06/11 16:25:08 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][10/29]  lr: 3.8442e-04  eta: 0:09:04  time: 1.0372  data_time: 0.8424  memory: 2602  loss: 1.9017  loss_cls: 0.8520  loss_bbox: 1.0497
    06/11 16:25:08 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][11/29]  lr: 3.8842e-04  eta: 0:08:59  time: 1.0322  data_time: 0.8380  memory: 2627  loss: 1.8977  loss_cls: 0.8554  loss_bbox: 1.0423
    06/11 16:25:09 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][12/29]  lr: 3.9243e-04  eta: 0:08:53  time: 1.0313  data_time: 0.8372  memory: 2612  loss: 1.8904  loss_cls: 0.8550  loss_bbox: 1.0355
    06/11 16:25:11 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][13/29]  lr: 3.9643e-04  eta: 0:08:58  time: 1.0332  data_time: 0.8376  memory: 2611  loss: 1.8843  loss_cls: 0.8589  loss_bbox: 1.0255
    06/11 16:25:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][14/29]  lr: 4.0044e-04  eta: 0:08:56  time: 1.0446  data_time: 0.8474  memory: 2620  loss: 1.8827  loss_cls: 0.8624  loss_bbox: 1.0203
    06/11 16:25:13 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][15/29]  lr: 4.0444e-04  eta: 0:08:55  time: 1.0461  data_time: 0.8481  memory: 2617  loss: 1.8755  loss_cls: 0.8656  loss_bbox: 1.0099
    06/11 16:25:13 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][16/29]  lr: 4.0844e-04  eta: 0:08:49  time: 1.0440  data_time: 0.8465  memory: 2605  loss: 1.8693  loss_cls: 0.8687  loss_bbox: 1.0006
    06/11 16:25:15 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][17/29]  lr: 4.1245e-04  eta: 0:08:51  time: 1.0237  data_time: 0.8273  memory: 2612  loss: 1.8596  loss_cls: 0.8659  loss_bbox: 0.9936
    06/11 16:25:15 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][18/29]  lr: 4.1645e-04  eta: 0:08:48  time: 1.0298  data_time: 0.8349  memory: 2605  loss: 1.8548  loss_cls: 0.8686  loss_bbox: 0.9862
    06/11 16:25:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][19/29]  lr: 4.2046e-04  eta: 0:08:49  time: 1.0520  data_time: 0.8559  memory: 2624  loss: 1.8463  loss_cls: 0.8644  loss_bbox: 0.9819
    06/11 16:25:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][20/29]  lr: 4.2446e-04  eta: 0:08:44  time: 1.0534  data_time: 0.8568  memory: 2602  loss: 1.8431  loss_cls: 0.8679  loss_bbox: 0.9752
    06/11 16:25:19 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][21/29]  lr: 4.2846e-04  eta: 0:08:46  time: 1.0755  data_time: 0.8787  memory: 2618  loss: 1.8322  loss_cls: 0.8628  loss_bbox: 0.9694
    06/11 16:25:19 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][22/29]  lr: 4.3247e-04  eta: 0:08:41  time: 0.9981  data_time: 0.8034  memory: 2656  loss: 1.8274  loss_cls: 0.8622  loss_bbox: 0.9652
    06/11 16:25:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][23/29]  lr: 4.3647e-04  eta: 0:08:43  time: 1.0332  data_time: 0.8375  memory: 2611  loss: 1.8192  loss_cls: 0.8613  loss_bbox: 0.9579
    06/11 16:25:22 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][24/29]  lr: 4.4048e-04  eta: 0:08:39  time: 1.0322  data_time: 0.8366  memory: 2615  loss: 1.8146  loss_cls: 0.8622  loss_bbox: 0.9524
    06/11 16:25:23 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][25/29]  lr: 4.4448e-04  eta: 0:08:41  time: 1.0661  data_time: 0.8704  memory: 2615  loss: 1.8078  loss_cls: 0.8616  loss_bbox: 0.9463
    06/11 16:25:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][26/29]  lr: 4.4848e-04  eta: 0:08:36  time: 1.0095  data_time: 0.8149  memory: 2636  loss: 1.8043  loss_cls: 0.8623  loss_bbox: 0.9421
    06/11 16:25:25 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][27/29]  lr: 4.5249e-04  eta: 0:08:38  time: 1.0378  data_time: 0.8416  memory: 2612  loss: 1.7960  loss_cls: 0.8617  loss_bbox: 0.9343
    06/11 16:25:26 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][28/29]  lr: 4.5649e-04  eta: 0:08:33  time: 1.0382  data_time: 0.8417  memory: 2605  loss: 1.7905  loss_cls: 0.8641  loss_bbox: 0.9264
    06/11 16:25:26 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:25:26 - mmengine - [4m[97mINFO[0m - Epoch(train)  [4][29/29]  lr: 4.6050e-04  eta: 0:08:28  time: 1.0309  data_time: 0.8360  memory: 1360  loss: 1.7796  loss_cls: 0.8574  loss_bbox: 0.9222
    06/11 16:25:30 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 1/29]  lr: 4.6450e-04  eta: 0:08:41  time: 1.0641  data_time: 0.8669  memory: 2614  loss: 1.7722  loss_cls: 0.8564  loss_bbox: 0.9158
    06/11 16:25:31 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 2/29]  lr: 4.6850e-04  eta: 0:08:37  time: 1.0407  data_time: 0.8426  memory: 2606  loss: 1.7649  loss_cls: 0.8550  loss_bbox: 0.9100
    06/11 16:25:31 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 3/29]  lr: 4.7251e-04  eta: 0:08:32  time: 1.0408  data_time: 0.8426  memory: 2609  loss: 1.7587  loss_cls: 0.8561  loss_bbox: 0.9025
    06/11 16:25:31 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 4/29]  lr: 4.7651e-04  eta: 0:08:28  time: 1.0407  data_time: 0.8426  memory: 2608  loss: 1.7520  loss_cls: 0.8552  loss_bbox: 0.8969
    06/11 16:25:35 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 5/29]  lr: 4.8052e-04  eta: 0:08:35  time: 1.0706  data_time: 0.8719  memory: 2619  loss: 1.7475  loss_cls: 0.8554  loss_bbox: 0.8922
    06/11 16:25:35 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 6/29]  lr: 4.8452e-04  eta: 0:08:31  time: 1.0486  data_time: 0.8503  memory: 2615  loss: 1.7406  loss_cls: 0.8549  loss_bbox: 0.8857
    06/11 16:25:35 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 7/29]  lr: 4.8852e-04  eta: 0:08:27  time: 1.0511  data_time: 0.8521  memory: 2610  loss: 1.7365  loss_cls: 0.8541  loss_bbox: 0.8824
    06/11 16:25:35 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 8/29]  lr: 4.9253e-04  eta: 0:08:22  time: 1.0513  data_time: 0.8522  memory: 2611  loss: 1.7298  loss_cls: 0.8554  loss_bbox: 0.8745
    06/11 16:25:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][ 9/29]  lr: 4.9653e-04  eta: 0:08:29  time: 1.0909  data_time: 0.8917  memory: 2610  loss: 1.7232  loss_cls: 0.8528  loss_bbox: 0.8704
    06/11 16:25:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][10/29]  lr: 5.0054e-04  eta: 0:08:24  time: 1.0440  data_time: 0.8454  memory: 2610  loss: 1.7190  loss_cls: 0.8544  loss_bbox: 0.8645
    06/11 16:25:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][11/29]  lr: 5.0454e-04  eta: 0:08:20  time: 1.0429  data_time: 0.8446  memory: 2629  loss: 1.7091  loss_cls: 0.8501  loss_bbox: 0.8590
    06/11 16:25:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][12/29]  lr: 5.0854e-04  eta: 0:08:15  time: 1.0420  data_time: 0.8439  memory: 2604  loss: 1.6966  loss_cls: 0.8433  loss_bbox: 0.8533
    06/11 16:25:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][13/29]  lr: 5.1255e-04  eta: 0:08:24  time: 1.1052  data_time: 0.9047  memory: 2617  loss: 1.6864  loss_cls: 0.8371  loss_bbox: 0.8493
    06/11 16:25:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][14/29]  lr: 5.1655e-04  eta: 0:08:20  time: 1.0557  data_time: 0.8551  memory: 2643  loss: 1.6780  loss_cls: 0.8347  loss_bbox: 0.8433
    06/11 16:25:44 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][15/29]  lr: 5.2056e-04  eta: 0:08:16  time: 1.0564  data_time: 0.8556  memory: 2609  loss: 1.6658  loss_cls: 0.8288  loss_bbox: 0.8370
    06/11 16:25:44 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][16/29]  lr: 5.2456e-04  eta: 0:08:12  time: 1.0566  data_time: 0.8552  memory: 2623  loss: 1.6590  loss_cls: 0.8283  loss_bbox: 0.8307
    06/11 16:25:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][17/29]  lr: 5.2856e-04  eta: 0:08:18  time: 1.1091  data_time: 0.9073  memory: 2611  loss: 1.6515  loss_cls: 0.8265  loss_bbox: 0.8250
    06/11 16:25:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][18/29]  lr: 5.3257e-04  eta: 0:08:14  time: 1.0554  data_time: 0.8536  memory: 2602  loss: 1.6560  loss_cls: 0.8286  loss_bbox: 0.8274
    06/11 16:25:48 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][19/29]  lr: 5.3657e-04  eta: 0:08:10  time: 1.0560  data_time: 0.8535  memory: 2634  loss: 1.6498  loss_cls: 0.8279  loss_bbox: 0.8219
    06/11 16:25:48 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][20/29]  lr: 5.4058e-04  eta: 0:08:07  time: 1.0586  data_time: 0.8537  memory: 2609  loss: 1.6430  loss_cls: 0.8250  loss_bbox: 0.8179
    06/11 16:25:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][21/29]  lr: 5.4458e-04  eta: 0:08:11  time: 1.1088  data_time: 0.9012  memory: 2613  loss: 1.6394  loss_cls: 0.8260  loss_bbox: 0.8134
    06/11 16:25:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][22/29]  lr: 5.4858e-04  eta: 0:08:07  time: 1.0268  data_time: 0.8207  memory: 2619  loss: 1.6355  loss_cls: 0.8244  loss_bbox: 0.8110
    06/11 16:25:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][23/29]  lr: 5.5259e-04  eta: 0:08:03  time: 1.0266  data_time: 0.8205  memory: 2610  loss: 1.6286  loss_cls: 0.8222  loss_bbox: 0.8063
    06/11 16:25:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][24/29]  lr: 5.5659e-04  eta: 0:07:59  time: 1.0268  data_time: 0.8209  memory: 2603  loss: 1.6178  loss_cls: 0.8165  loss_bbox: 0.8013
    06/11 16:25:54 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][25/29]  lr: 5.6059e-04  eta: 0:08:03  time: 1.0736  data_time: 0.8661  memory: 2636  loss: 1.6116  loss_cls: 0.8141  loss_bbox: 0.7975
    06/11 16:25:54 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][26/29]  lr: 5.6460e-04  eta: 0:07:59  time: 1.0156  data_time: 0.8084  memory: 2628  loss: 1.6034  loss_cls: 0.8094  loss_bbox: 0.7939
    06/11 16:25:54 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][27/29]  lr: 5.6860e-04  eta: 0:07:55  time: 1.0079  data_time: 0.8014  memory: 2625  loss: 1.5949  loss_cls: 0.8043  loss_bbox: 0.7906
    06/11 16:25:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][28/29]  lr: 5.7261e-04  eta: 0:07:51  time: 1.0077  data_time: 0.8014  memory: 2612  loss: 1.5872  loss_cls: 0.8023  loss_bbox: 0.7849
    06/11 16:25:56 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:25:56 - mmengine - [4m[97mINFO[0m - Epoch(train)  [5][29/29]  lr: 5.7661e-04  eta: 0:07:52  time: 1.0209  data_time: 0.8157  memory: 1402  loss: 1.5840  loss_cls: 0.8008  loss_bbox: 0.7832
    06/11 16:25:59 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 1/14]    eta: 0:00:35  time: 2.7430  data_time: 2.6017  memory: 245  
    06/11 16:25:59 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 2/14]    eta: 0:00:17  time: 1.4365  data_time: 1.3172  memory: 245  
    06/11 16:26:00 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 3/14]    eta: 0:00:14  time: 1.3201  data_time: 1.1963  memory: 245  
    06/11 16:26:00 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 4/14]    eta: 0:00:10  time: 1.0194  data_time: 0.9005  memory: 245  
    06/11 16:26:01 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 5/14]    eta: 0:00:09  time: 1.0402  data_time: 0.9255  memory: 245  
    06/11 16:26:01 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 6/14]    eta: 0:00:07  time: 0.8818  data_time: 0.7718  memory: 245  
    06/11 16:26:03 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 7/14]    eta: 0:00:06  time: 0.9448  data_time: 0.8360  memory: 245  
    06/11 16:26:03 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 8/14]    eta: 0:00:05  time: 0.8379  data_time: 0.7320  memory: 245  
    06/11 16:26:04 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][ 9/14]    eta: 0:00:04  time: 0.8592  data_time: 0.7542  memory: 245  
    06/11 16:26:04 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][10/14]    eta: 0:00:03  time: 0.7969  data_time: 0.6934  memory: 245  
    06/11 16:26:05 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][11/14]    eta: 0:00:02  time: 0.8083  data_time: 0.7005  memory: 245  
    06/11 16:26:06 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][12/14]    eta: 0:00:01  time: 0.8015  data_time: 0.6902  memory: 245  
    06/11 16:26:06 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][13/14]    eta: 0:00:00  time: 0.7701  data_time: 0.6568  memory: 245  
    06/11 16:26:07 - mmengine - [4m[97mINFO[0m - Epoch(val)  [5][14/14]    eta: 0:00:00  time: 0.7805  data_time: 0.6645  memory: 245  
    06/11 16:26:07 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.60s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.029
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.093
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.006
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.029
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.049
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.322
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
    06/11 16:26:08 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.029 0.093 0.006 -1.000 -1.000 0.029
    06/11 16:26:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [5][14/14]    coco/bbox_mAP: 0.0290  coco/bbox_mAP_50: 0.0930  coco/bbox_mAP_75: 0.0060  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.0290  data_time: 0.6645  time: 0.7805
    06/11 16:26:13 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.0290 coco/bbox_mAP at 5 epoch is saved to best_coco_bbox_mAP_epoch_5.pth.
    06/11 16:26:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 1/29]  lr: 5.8061e-04  eta: 0:07:59  time: 1.0591  data_time: 0.8524  memory: 2604  loss: 1.5759  loss_cls: 0.7973  loss_bbox: 0.7785
    06/11 16:26:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 2/29]  lr: 5.8462e-04  eta: 0:07:55  time: 1.0439  data_time: 0.8381  memory: 2616  loss: 1.5718  loss_cls: 0.7945  loss_bbox: 0.7772
    06/11 16:26:18 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 3/29]  lr: 5.8862e-04  eta: 0:07:54  time: 1.0560  data_time: 0.8489  memory: 2646  loss: 1.5646  loss_cls: 0.7911  loss_bbox: 0.7735
    06/11 16:26:18 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 4/29]  lr: 5.9263e-04  eta: 0:07:50  time: 1.0547  data_time: 0.8476  memory: 2651  loss: 1.5607  loss_cls: 0.7891  loss_bbox: 0.7715
    06/11 16:26:20 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 5/29]  lr: 5.9663e-04  eta: 0:07:50  time: 1.0390  data_time: 0.8342  memory: 2602  loss: 1.5557  loss_cls: 0.7868  loss_bbox: 0.7689
    06/11 16:26:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 6/29]  lr: 6.0063e-04  eta: 0:07:49  time: 1.0408  data_time: 0.8373  memory: 2640  loss: 1.5495  loss_cls: 0.7844  loss_bbox: 0.7652
    06/11 16:26:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 7/29]  lr: 6.0464e-04  eta: 0:07:47  time: 1.0343  data_time: 0.8315  memory: 2607  loss: 1.5450  loss_cls: 0.7820  loss_bbox: 0.7630
    06/11 16:26:22 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 8/29]  lr: 6.0864e-04  eta: 0:07:43  time: 1.0346  data_time: 0.8317  memory: 2618  loss: 1.5418  loss_cls: 0.7809  loss_bbox: 0.7609
    06/11 16:26:23 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][ 9/29]  lr: 6.1265e-04  eta: 0:07:43  time: 1.0278  data_time: 0.8250  memory: 2609  loss: 1.5378  loss_cls: 0.7783  loss_bbox: 0.7595
    06/11 16:26:25 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][10/29]  lr: 6.1665e-04  eta: 0:07:44  time: 1.0545  data_time: 0.8504  memory: 2619  loss: 1.5326  loss_cls: 0.7756  loss_bbox: 0.7571
    06/11 16:26:26 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][11/29]  lr: 6.2065e-04  eta: 0:07:42  time: 1.0385  data_time: 0.8346  memory: 2615  loss: 1.5271  loss_cls: 0.7762  loss_bbox: 0.7509
    06/11 16:26:26 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][12/29]  lr: 6.2466e-04  eta: 0:07:39  time: 1.0376  data_time: 0.8342  memory: 2605  loss: 1.5247  loss_cls: 0.7733  loss_bbox: 0.7514
    06/11 16:26:27 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][13/29]  lr: 6.2866e-04  eta: 0:07:38  time: 1.0259  data_time: 0.8209  memory: 2615  loss: 1.5227  loss_cls: 0.7730  loss_bbox: 0.7497
    06/11 16:26:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][14/29]  lr: 6.3267e-04  eta: 0:07:40  time: 1.0665  data_time: 0.8583  memory: 2612  loss: 1.5178  loss_cls: 0.7706  loss_bbox: 0.7471
    06/11 16:26:30 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][15/29]  lr: 6.3667e-04  eta: 0:07:37  time: 1.0321  data_time: 0.8234  memory: 2602  loss: 1.5150  loss_cls: 0.7709  loss_bbox: 0.7441
    06/11 16:26:30 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][16/29]  lr: 6.4067e-04  eta: 0:07:34  time: 1.0375  data_time: 0.8270  memory: 2616  loss: 1.5098  loss_cls: 0.7665  loss_bbox: 0.7434
    06/11 16:26:31 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][17/29]  lr: 6.4468e-04  eta: 0:07:33  time: 1.0182  data_time: 0.8080  memory: 2630  loss: 1.5057  loss_cls: 0.7631  loss_bbox: 0.7426
    06/11 16:26:33 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][18/29]  lr: 6.4868e-04  eta: 0:07:34  time: 1.0536  data_time: 0.8426  memory: 2609  loss: 1.5007  loss_cls: 0.7630  loss_bbox: 0.7377
    06/11 16:26:34 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][19/29]  lr: 6.5269e-04  eta: 0:07:31  time: 1.0264  data_time: 0.8169  memory: 2610  loss: 1.4984  loss_cls: 0.7616  loss_bbox: 0.7368
    06/11 16:26:34 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][20/29]  lr: 6.5669e-04  eta: 0:07:28  time: 1.0264  data_time: 0.8170  memory: 2618  loss: 1.4948  loss_cls: 0.7602  loss_bbox: 0.7346
    06/11 16:26:35 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][21/29]  lr: 6.6069e-04  eta: 0:07:27  time: 1.0488  data_time: 0.8366  memory: 2621  loss: 1.4936  loss_cls: 0.7611  loss_bbox: 0.7324
    06/11 16:26:37 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][22/29]  lr: 6.6470e-04  eta: 0:07:28  time: 0.9947  data_time: 0.7829  memory: 2602  loss: 1.4914  loss_cls: 0.7609  loss_bbox: 0.7305
    06/11 16:26:37 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][23/29]  lr: 6.6870e-04  eta: 0:07:25  time: 0.9910  data_time: 0.7804  memory: 2622  loss: 1.4904  loss_cls: 0.7630  loss_bbox: 0.7274
    06/11 16:26:37 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][24/29]  lr: 6.7271e-04  eta: 0:07:22  time: 0.9910  data_time: 0.7796  memory: 2617  loss: 1.4877  loss_cls: 0.7618  loss_bbox: 0.7259
    06/11 16:26:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][25/29]  lr: 6.7671e-04  eta: 0:07:22  time: 1.0199  data_time: 0.8066  memory: 2605  loss: 1.4893  loss_cls: 0.7642  loss_bbox: 0.7251
    06/11 16:26:41 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][26/29]  lr: 6.8071e-04  eta: 0:07:23  time: 0.9922  data_time: 0.7778  memory: 2617  loss: 1.4837  loss_cls: 0.7622  loss_bbox: 0.7215
    06/11 16:26:41 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][27/29]  lr: 6.8472e-04  eta: 0:07:20  time: 0.9942  data_time: 0.7789  memory: 2615  loss: 1.4817  loss_cls: 0.7639  loss_bbox: 0.7178
    06/11 16:26:42 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][28/29]  lr: 6.8872e-04  eta: 0:07:17  time: 0.9916  data_time: 0.7771  memory: 2602  loss: 1.4795  loss_cls: 0.7632  loss_bbox: 0.7162
    06/11 16:26:42 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:26:42 - mmengine - [4m[97mINFO[0m - Epoch(train)  [6][29/29]  lr: 6.9273e-04  eta: 0:07:14  time: 0.9907  data_time: 0.7770  memory: 1362  loss: 1.4773  loss_cls: 0.7635  loss_bbox: 0.7137
    06/11 16:26:46 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 1/29]  lr: 6.9673e-04  eta: 0:07:20  time: 1.0094  data_time: 0.7953  memory: 2607  loss: 1.4734  loss_cls: 0.7631  loss_bbox: 0.7103
    06/11 16:26:46 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 2/29]  lr: 7.0073e-04  eta: 0:07:17  time: 1.0097  data_time: 0.7954  memory: 2617  loss: 1.4702  loss_cls: 0.7626  loss_bbox: 0.7077
    06/11 16:26:46 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 3/29]  lr: 7.0474e-04  eta: 0:07:14  time: 1.0103  data_time: 0.7958  memory: 2641  loss: 1.4693  loss_cls: 0.7631  loss_bbox: 0.7062
    06/11 16:26:46 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 4/29]  lr: 7.0874e-04  eta: 0:07:11  time: 1.0107  data_time: 0.7961  memory: 2617  loss: 1.4702  loss_cls: 0.7661  loss_bbox: 0.7041
    06/11 16:26:49 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 5/29]  lr: 7.1275e-04  eta: 0:07:14  time: 0.9924  data_time: 0.7782  memory: 2619  loss: 1.4704  loss_cls: 0.7684  loss_bbox: 0.7019
    06/11 16:26:50 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 6/29]  lr: 7.1675e-04  eta: 0:07:13  time: 1.0069  data_time: 0.7923  memory: 2602  loss: 1.4677  loss_cls: 0.7671  loss_bbox: 0.7006
    06/11 16:26:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 7/29]  lr: 7.2075e-04  eta: 0:07:10  time: 1.0083  data_time: 0.7919  memory: 2633  loss: 1.4681  loss_cls: 0.7671  loss_bbox: 0.7010
    06/11 16:26:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 8/29]  lr: 7.2476e-04  eta: 0:07:07  time: 1.0082  data_time: 0.7918  memory: 2632  loss: 1.4668  loss_cls: 0.7657  loss_bbox: 0.7011
    06/11 16:26:54 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][ 9/29]  lr: 7.2876e-04  eta: 0:07:10  time: 0.9961  data_time: 0.7779  memory: 2627  loss: 1.4647  loss_cls: 0.7639  loss_bbox: 0.7007
    06/11 16:26:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][10/29]  lr: 7.3277e-04  eta: 0:07:08  time: 1.0121  data_time: 0.7921  memory: 2607  loss: 1.4487  loss_cls: 0.7561  loss_bbox: 0.6926
    06/11 16:26:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][11/29]  lr: 7.3677e-04  eta: 0:07:05  time: 1.0112  data_time: 0.7914  memory: 2602  loss: 1.4465  loss_cls: 0.7561  loss_bbox: 0.6904
    06/11 16:26:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][12/29]  lr: 7.4077e-04  eta: 0:07:02  time: 1.0083  data_time: 0.7907  memory: 2604  loss: 1.4414  loss_cls: 0.7529  loss_bbox: 0.6885
    06/11 16:26:58 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][13/29]  lr: 7.4478e-04  eta: 0:07:04  time: 1.0052  data_time: 0.7869  memory: 2633  loss: 1.4344  loss_cls: 0.7479  loss_bbox: 0.6865
    06/11 16:26:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][14/29]  lr: 7.4878e-04  eta: 0:07:03  time: 1.0229  data_time: 0.8038  memory: 2606  loss: 1.4285  loss_cls: 0.7457  loss_bbox: 0.6828
    06/11 16:26:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][15/29]  lr: 7.5279e-04  eta: 0:07:01  time: 1.0236  data_time: 0.8041  memory: 2625  loss: 1.4248  loss_cls: 0.7426  loss_bbox: 0.6823
    06/11 16:26:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][16/29]  lr: 7.5679e-04  eta: 0:06:58  time: 1.0236  data_time: 0.8035  memory: 2651  loss: 1.4239  loss_cls: 0.7407  loss_bbox: 0.6832
    06/11 16:27:02 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][17/29]  lr: 7.6079e-04  eta: 0:07:00  time: 1.0206  data_time: 0.8021  memory: 2618  loss: 1.4199  loss_cls: 0.7379  loss_bbox: 0.6820
    06/11 16:27:02 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][18/29]  lr: 7.6480e-04  eta: 0:06:57  time: 1.0208  data_time: 0.8023  memory: 2612  loss: 1.4165  loss_cls: 0.7355  loss_bbox: 0.6811
    06/11 16:27:02 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][19/29]  lr: 7.6880e-04  eta: 0:06:54  time: 1.0219  data_time: 0.8028  memory: 2637  loss: 1.4143  loss_cls: 0.7348  loss_bbox: 0.6795
    06/11 16:27:03 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][20/29]  lr: 7.7281e-04  eta: 0:06:52  time: 1.0335  data_time: 0.8124  memory: 2629  loss: 1.4118  loss_cls: 0.7314  loss_bbox: 0.6804
    06/11 16:27:06 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][21/29]  lr: 7.7681e-04  eta: 0:06:54  time: 1.0536  data_time: 0.8314  memory: 2622  loss: 1.4050  loss_cls: 0.7285  loss_bbox: 0.6765
    06/11 16:27:06 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][22/29]  lr: 7.8081e-04  eta: 0:06:52  time: 0.9816  data_time: 0.7617  memory: 2623  loss: 1.4026  loss_cls: 0.7272  loss_bbox: 0.6753
    06/11 16:27:06 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][23/29]  lr: 7.8482e-04  eta: 0:06:49  time: 0.9808  data_time: 0.7619  memory: 2612  loss: 1.3989  loss_cls: 0.7257  loss_bbox: 0.6732
    06/11 16:27:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][24/29]  lr: 7.8882e-04  eta: 0:06:47  time: 0.9776  data_time: 0.7600  memory: 2602  loss: 1.3945  loss_cls: 0.7247  loss_bbox: 0.6698
    06/11 16:27:10 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][25/29]  lr: 7.9282e-04  eta: 0:06:50  time: 1.0322  data_time: 0.8127  memory: 2604  loss: 1.3902  loss_cls: 0.7228  loss_bbox: 0.6674
    06/11 16:27:10 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][26/29]  lr: 7.9683e-04  eta: 0:06:47  time: 1.0057  data_time: 0.7848  memory: 2636  loss: 1.3879  loss_cls: 0.7211  loss_bbox: 0.6668
    06/11 16:27:10 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][27/29]  lr: 8.0083e-04  eta: 0:06:44  time: 0.9933  data_time: 0.7726  memory: 2633  loss: 1.3832  loss_cls: 0.7191  loss_bbox: 0.6641
    06/11 16:27:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][28/29]  lr: 8.0484e-04  eta: 0:06:44  time: 1.0020  data_time: 0.7809  memory: 2603  loss: 1.3811  loss_cls: 0.7194  loss_bbox: 0.6617
    06/11 16:27:12 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:27:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [7][29/29]  lr: 8.0884e-04  eta: 0:06:41  time: 1.0036  data_time: 0.7827  memory: 1345  loss: 1.3752  loss_cls: 0.7182  loss_bbox: 0.6570
    06/11 16:27:16 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 1/29]  lr: 8.1284e-04  eta: 0:06:46  time: 1.0567  data_time: 0.8358  memory: 2602  loss: 1.3725  loss_cls: 0.7194  loss_bbox: 0.6531
    06/11 16:27:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 2/29]  lr: 8.1685e-04  eta: 0:06:45  time: 1.0384  data_time: 0.8185  memory: 2619  loss: 1.3679  loss_cls: 0.7194  loss_bbox: 0.6485
    06/11 16:27:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 3/29]  lr: 8.2085e-04  eta: 0:06:42  time: 1.0280  data_time: 0.8095  memory: 2602  loss: 1.3637  loss_cls: 0.7190  loss_bbox: 0.6447
    06/11 16:27:17 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 4/29]  lr: 8.2486e-04  eta: 0:06:39  time: 1.0276  data_time: 0.8092  memory: 2625  loss: 1.3564  loss_cls: 0.7163  loss_bbox: 0.6401
    06/11 16:27:20 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 5/29]  lr: 8.2886e-04  eta: 0:06:40  time: 1.0450  data_time: 0.8258  memory: 2616  loss: 1.3519  loss_cls: 0.7149  loss_bbox: 0.6370
    06/11 16:27:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 6/29]  lr: 8.3286e-04  eta: 0:06:40  time: 1.0299  data_time: 0.8140  memory: 2631  loss: 1.3468  loss_cls: 0.7132  loss_bbox: 0.6337
    06/11 16:27:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 7/29]  lr: 8.3687e-04  eta: 0:06:37  time: 1.0286  data_time: 0.8146  memory: 2614  loss: 1.3419  loss_cls: 0.7114  loss_bbox: 0.6305
    06/11 16:27:21 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 8/29]  lr: 8.4087e-04  eta: 0:06:34  time: 1.0228  data_time: 0.8112  memory: 2611  loss: 1.3394  loss_cls: 0.7123  loss_bbox: 0.6271
    06/11 16:27:24 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][ 9/29]  lr: 8.4488e-04  eta: 0:06:35  time: 1.0453  data_time: 0.8331  memory: 2619  loss: 1.3364  loss_cls: 0.7122  loss_bbox: 0.6243
    06/11 16:27:25 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][10/29]  lr: 8.4888e-04  eta: 0:06:35  time: 1.0313  data_time: 0.8197  memory: 2637  loss: 1.3323  loss_cls: 0.7086  loss_bbox: 0.6237
    06/11 16:27:25 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][11/29]  lr: 8.5288e-04  eta: 0:06:32  time: 1.0283  data_time: 0.8172  memory: 2610  loss: 1.3272  loss_cls: 0.7084  loss_bbox: 0.6188
    06/11 16:27:25 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][12/29]  lr: 8.5689e-04  eta: 0:06:29  time: 1.0270  data_time: 0.8165  memory: 2607  loss: 1.3234  loss_cls: 0.7060  loss_bbox: 0.6174
    06/11 16:27:27 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][13/29]  lr: 8.6089e-04  eta: 0:06:29  time: 1.0360  data_time: 0.8259  memory: 2617  loss: 1.3213  loss_cls: 0.7057  loss_bbox: 0.6156
    06/11 16:27:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][14/29]  lr: 8.6490e-04  eta: 0:06:30  time: 1.0398  data_time: 0.8305  memory: 2603  loss: 1.3198  loss_cls: 0.7042  loss_bbox: 0.6156
    06/11 16:27:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][15/29]  lr: 8.6890e-04  eta: 0:06:28  time: 1.0403  data_time: 0.8310  memory: 2604  loss: 1.3160  loss_cls: 0.7038  loss_bbox: 0.6122
    06/11 16:27:29 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][16/29]  lr: 8.7290e-04  eta: 0:06:25  time: 1.0395  data_time: 0.8313  memory: 2616  loss: 1.3121  loss_cls: 0.7024  loss_bbox: 0.6096
    06/11 16:27:31 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][17/29]  lr: 8.7691e-04  eta: 0:06:25  time: 1.0366  data_time: 0.8291  memory: 2615  loss: 1.3052  loss_cls: 0.6992  loss_bbox: 0.6060
    06/11 16:27:32 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][18/29]  lr: 8.8091e-04  eta: 0:06:24  time: 1.0205  data_time: 0.8150  memory: 2610  loss: 1.3054  loss_cls: 0.6995  loss_bbox: 0.6059
    06/11 16:27:32 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][19/29]  lr: 8.8492e-04  eta: 0:06:21  time: 1.0186  data_time: 0.8137  memory: 2614  loss: 1.3022  loss_cls: 0.6963  loss_bbox: 0.6059
    06/11 16:27:34 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][20/29]  lr: 8.8892e-04  eta: 0:06:21  time: 1.0426  data_time: 0.8362  memory: 2614  loss: 1.2979  loss_cls: 0.6956  loss_bbox: 0.6023
    06/11 16:27:35 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][21/29]  lr: 8.9292e-04  eta: 0:06:20  time: 1.0604  data_time: 0.8520  memory: 2633  loss: 1.2959  loss_cls: 0.6949  loss_bbox: 0.6010
    06/11 16:27:36 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][22/29]  lr: 8.9693e-04  eta: 0:06:19  time: 0.9987  data_time: 0.7904  memory: 2616  loss: 1.2915  loss_cls: 0.6925  loss_bbox: 0.5990
    06/11 16:27:36 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][23/29]  lr: 9.0093e-04  eta: 0:06:16  time: 0.9982  data_time: 0.7903  memory: 2613  loss: 1.2881  loss_cls: 0.6904  loss_bbox: 0.5976
    06/11 16:27:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][24/29]  lr: 9.0494e-04  eta: 0:06:18  time: 1.0480  data_time: 0.8387  memory: 2659  loss: 1.2864  loss_cls: 0.6891  loss_bbox: 0.5973
    06/11 16:27:39 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][25/29]  lr: 9.0894e-04  eta: 0:06:15  time: 1.0486  data_time: 0.8388  memory: 2625  loss: 1.2822  loss_cls: 0.6865  loss_bbox: 0.5957
    06/11 16:27:40 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][26/29]  lr: 9.1294e-04  eta: 0:06:14  time: 1.0037  data_time: 0.7942  memory: 2627  loss: 1.2775  loss_cls: 0.6827  loss_bbox: 0.5948
    06/11 16:27:40 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][27/29]  lr: 9.1695e-04  eta: 0:06:11  time: 0.9884  data_time: 0.7792  memory: 2612  loss: 1.2731  loss_cls: 0.6804  loss_bbox: 0.5927
    06/11 16:27:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][28/29]  lr: 9.2095e-04  eta: 0:06:13  time: 1.0340  data_time: 0.8250  memory: 2612  loss: 1.2697  loss_cls: 0.6786  loss_bbox: 0.5911
    06/11 16:27:43 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:27:43 - mmengine - [4m[97mINFO[0m - Epoch(train)  [8][29/29]  lr: 9.2496e-04  eta: 0:06:10  time: 1.0350  data_time: 0.8264  memory: 1374  loss: 1.2699  loss_cls: 0.6778  loss_bbox: 0.5922
    06/11 16:27:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 1/29]  lr: 9.2896e-04  eta: 0:06:14  time: 1.0617  data_time: 0.8537  memory: 2645  loss: 1.2692  loss_cls: 0.6771  loss_bbox: 0.5921
    06/11 16:27:47 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 2/29]  lr: 9.3296e-04  eta: 0:06:12  time: 1.0486  data_time: 0.8412  memory: 2621  loss: 1.2683  loss_cls: 0.6775  loss_bbox: 0.5908
    06/11 16:27:48 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 3/29]  lr: 9.3697e-04  eta: 0:06:09  time: 1.0495  data_time: 0.8415  memory: 2620  loss: 1.2638  loss_cls: 0.6753  loss_bbox: 0.5885
    06/11 16:27:48 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 4/29]  lr: 9.4097e-04  eta: 0:06:07  time: 1.0512  data_time: 0.8427  memory: 2604  loss: 1.2637  loss_cls: 0.6764  loss_bbox: 0.5873
    06/11 16:27:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 5/29]  lr: 9.4498e-04  eta: 0:06:09  time: 1.0563  data_time: 0.8495  memory: 2616  loss: 1.2616  loss_cls: 0.6762  loss_bbox: 0.5854
    06/11 16:27:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 6/29]  lr: 9.4898e-04  eta: 0:06:07  time: 1.0445  data_time: 0.8384  memory: 2637  loss: 1.2594  loss_cls: 0.6758  loss_bbox: 0.5836
    06/11 16:27:51 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 7/29]  lr: 9.5298e-04  eta: 0:06:05  time: 1.0457  data_time: 0.8399  memory: 2633  loss: 1.2556  loss_cls: 0.6739  loss_bbox: 0.5817
    06/11 16:27:52 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 8/29]  lr: 9.5699e-04  eta: 0:06:03  time: 1.0509  data_time: 0.8449  memory: 2605  loss: 1.2499  loss_cls: 0.6733  loss_bbox: 0.5767
    06/11 16:27:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][ 9/29]  lr: 9.6099e-04  eta: 0:06:04  time: 1.0578  data_time: 0.8502  memory: 2605  loss: 1.2504  loss_cls: 0.6758  loss_bbox: 0.5746
    06/11 16:27:55 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][10/29]  lr: 9.6500e-04  eta: 0:06:02  time: 1.0573  data_time: 0.8497  memory: 2610  loss: 1.2491  loss_cls: 0.6753  loss_bbox: 0.5738
    06/11 16:27:56 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][11/29]  lr: 9.6900e-04  eta: 0:06:00  time: 1.0681  data_time: 0.8605  memory: 2613  loss: 1.2462  loss_cls: 0.6755  loss_bbox: 0.5708
    06/11 16:27:56 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][12/29]  lr: 9.7300e-04  eta: 0:05:59  time: 1.0639  data_time: 0.8580  memory: 2607  loss: 1.2422  loss_cls: 0.6753  loss_bbox: 0.5669
    06/11 16:27:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][13/29]  lr: 9.7701e-04  eta: 0:05:59  time: 1.0593  data_time: 0.8539  memory: 2610  loss: 1.2401  loss_cls: 0.6760  loss_bbox: 0.5641
    06/11 16:27:59 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][14/29]  lr: 9.8101e-04  eta: 0:05:57  time: 1.0582  data_time: 0.8531  memory: 2610  loss: 1.2386  loss_cls: 0.6760  loss_bbox: 0.5626
    06/11 16:28:00 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][15/29]  lr: 9.8502e-04  eta: 0:05:56  time: 1.0824  data_time: 0.8772  memory: 2610  loss: 1.2363  loss_cls: 0.6747  loss_bbox: 0.5615
    06/11 16:28:01 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][16/29]  lr: 9.8902e-04  eta: 0:05:54  time: 1.0754  data_time: 0.8709  memory: 2602  loss: 1.2340  loss_cls: 0.6739  loss_bbox: 0.5601
    06/11 16:28:02 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][17/29]  lr: 9.9302e-04  eta: 0:05:53  time: 1.0407  data_time: 0.8383  memory: 2616  loss: 1.2316  loss_cls: 0.6719  loss_bbox: 0.5597
    06/11 16:28:02 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][18/29]  lr: 9.9703e-04  eta: 0:05:51  time: 1.0422  data_time: 0.8406  memory: 2621  loss: 1.2290  loss_cls: 0.6718  loss_bbox: 0.5572
    06/11 16:28:04 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][19/29]  lr: 1.0010e-03  eta: 0:05:51  time: 1.0688  data_time: 0.8675  memory: 2608  loss: 1.2270  loss_cls: 0.6710  loss_bbox: 0.5560
    06/11 16:28:05 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][20/29]  lr: 1.0050e-03  eta: 0:05:50  time: 1.0624  data_time: 0.8607  memory: 2615  loss: 1.2252  loss_cls: 0.6709  loss_bbox: 0.5544
    06/11 16:28:06 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][21/29]  lr: 1.0090e-03  eta: 0:05:49  time: 1.0752  data_time: 0.8714  memory: 2630  loss: 1.2255  loss_cls: 0.6698  loss_bbox: 0.5557
    06/11 16:28:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][22/29]  lr: 1.0130e-03  eta: 0:05:47  time: 1.0113  data_time: 0.8065  memory: 2614  loss: 1.2231  loss_cls: 0.6681  loss_bbox: 0.5550
    06/11 16:28:07 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][23/29]  lr: 1.0170e-03  eta: 0:05:46  time: 1.0084  data_time: 0.8025  memory: 2607  loss: 1.2187  loss_cls: 0.6658  loss_bbox: 0.5529
    06/11 16:28:09 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][24/29]  lr: 1.0211e-03  eta: 0:05:45  time: 1.0336  data_time: 0.8259  memory: 2613  loss: 1.2178  loss_cls: 0.6649  loss_bbox: 0.5528
    06/11 16:28:10 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][25/29]  lr: 1.0251e-03  eta: 0:05:44  time: 1.0425  data_time: 0.8325  memory: 2625  loss: 1.2161  loss_cls: 0.6632  loss_bbox: 0.5529
    06/11 16:28:10 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][26/29]  lr: 1.0291e-03  eta: 0:05:42  time: 1.0162  data_time: 0.8080  memory: 2628  loss: 1.2148  loss_cls: 0.6625  loss_bbox: 0.5523
    06/11 16:28:11 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][27/29]  lr: 1.0331e-03  eta: 0:05:41  time: 1.0066  data_time: 0.7967  memory: 2628  loss: 1.2121  loss_cls: 0.6609  loss_bbox: 0.5512
    06/11 16:28:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][28/29]  lr: 1.0371e-03  eta: 0:05:40  time: 1.0212  data_time: 0.8111  memory: 2604  loss: 1.2125  loss_cls: 0.6615  loss_bbox: 0.5510
    06/11 16:28:12 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:28:12 - mmengine - [4m[97mINFO[0m - Epoch(train)  [9][29/29]  lr: 1.0411e-03  eta: 0:05:38  time: 1.0196  data_time: 0.8104  memory: 1351  loss: 1.2123  loss_cls: 0.6608  loss_bbox: 0.5515
    06/11 16:28:14 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 1/14]    eta: 0:00:13  time: 0.7998  data_time: 0.6866  memory: 245  
    06/11 16:28:14 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 2/14]    eta: 0:00:06  time: 0.7543  data_time: 0.6439  memory: 245  
    06/11 16:28:14 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 3/14]    eta: 0:00:07  time: 0.7580  data_time: 0.6485  memory: 245  
    06/11 16:28:15 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 4/14]    eta: 0:00:05  time: 0.7201  data_time: 0.6126  memory: 245  
    06/11 16:28:15 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 5/14]    eta: 0:00:05  time: 0.7243  data_time: 0.6156  memory: 245  
    06/11 16:28:15 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 6/14]    eta: 0:00:03  time: 0.6932  data_time: 0.5850  memory: 245  
    06/11 16:28:16 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 7/14]    eta: 0:00:03  time: 0.6989  data_time: 0.5914  memory: 245  
    06/11 16:28:16 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 8/14]    eta: 0:00:02  time: 0.6704  data_time: 0.5646  memory: 245  
    06/11 16:28:17 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][ 9/14]    eta: 0:00:02  time: 0.6731  data_time: 0.5677  memory: 245  
    06/11 16:28:17 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][10/14]    eta: 0:00:01  time: 0.6483  data_time: 0.5441  memory: 245  
    06/11 16:28:18 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][11/14]    eta: 0:00:01  time: 0.6466  data_time: 0.5429  memory: 245  
    06/11 16:28:18 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][12/14]    eta: 0:00:00  time: 0.6247  data_time: 0.5221  memory: 245  
    06/11 16:28:18 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][13/14]    eta: 0:00:00  time: 0.6256  data_time: 0.5239  memory: 245  
    06/11 16:28:19 - mmengine - [4m[97mINFO[0m - Epoch(val)  [9][14/14]    eta: 0:00:00  time: 0.6136  data_time: 0.5110  memory: 245  
    06/11 16:28:19 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.61s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.451
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.204
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.227
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.613
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.620
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.620
    06/11 16:28:20 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.227 0.451 0.204 -1.000 -1.000 0.227
    06/11 16:28:20 - mmengine - [4m[97mINFO[0m - Epoch(val) [9][14/14]    coco/bbox_mAP: 0.2270  coco/bbox_mAP_50: 0.4510  coco/bbox_mAP_75: 0.2040  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.2270  data_time: 0.3846  time: 0.4779
    06/11 16:28:20 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_5.pth is removed
    06/11 16:28:24 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.2270 coco/bbox_mAP at 9 epoch is saved to best_coco_bbox_mAP_epoch_9.pth.
    06/11 16:28:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 1/29]  lr: 1.0451e-03  eta: 0:05:41  time: 1.0655  data_time: 0.8556  memory: 2608  loss: 1.2117  loss_cls: 0.6597  loss_bbox: 0.5520
    06/11 16:28:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 2/29]  lr: 1.0491e-03  eta: 0:05:39  time: 1.0439  data_time: 0.8339  memory: 2638  loss: 1.2106  loss_cls: 0.6599  loss_bbox: 0.5506
    06/11 16:28:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 3/29]  lr: 1.0531e-03  eta: 0:05:37  time: 1.0448  data_time: 0.8349  memory: 2604  loss: 1.2096  loss_cls: 0.6595  loss_bbox: 0.5501
    06/11 16:28:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 4/29]  lr: 1.0571e-03  eta: 0:05:35  time: 1.0515  data_time: 0.8407  memory: 2609  loss: 1.2089  loss_cls: 0.6585  loss_bbox: 0.5504
    06/11 16:28:33 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 5/29]  lr: 1.0611e-03  eta: 0:05:37  time: 1.0864  data_time: 0.8757  memory: 2622  loss: 1.2073  loss_cls: 0.6576  loss_bbox: 0.5497
    06/11 16:28:33 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 6/29]  lr: 1.0651e-03  eta: 0:05:35  time: 1.0506  data_time: 0.8407  memory: 2612  loss: 1.2031  loss_cls: 0.6559  loss_bbox: 0.5472
    06/11 16:28:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 7/29]  lr: 1.0691e-03  eta: 0:05:33  time: 1.0497  data_time: 0.8400  memory: 2602  loss: 1.2012  loss_cls: 0.6541  loss_bbox: 0.5472
    06/11 16:28:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 8/29]  lr: 1.0731e-03  eta: 0:05:31  time: 1.0491  data_time: 0.8397  memory: 2606  loss: 1.1989  loss_cls: 0.6525  loss_bbox: 0.5464
    06/11 16:28:37 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][ 9/29]  lr: 1.0771e-03  eta: 0:05:32  time: 1.0862  data_time: 0.8766  memory: 2625  loss: 1.1990  loss_cls: 0.6529  loss_bbox: 0.5460
    06/11 16:28:37 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][10/29]  lr: 1.0811e-03  eta: 0:05:30  time: 1.0652  data_time: 0.8557  memory: 2602  loss: 1.1983  loss_cls: 0.6535  loss_bbox: 0.5448
    06/11 16:28:38 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][11/29]  lr: 1.0851e-03  eta: 0:05:28  time: 1.0662  data_time: 0.8564  memory: 2614  loss: 1.1976  loss_cls: 0.6527  loss_bbox: 0.5449
    06/11 16:28:38 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][12/29]  lr: 1.0891e-03  eta: 0:05:26  time: 1.0429  data_time: 0.8342  memory: 2620  loss: 1.1955  loss_cls: 0.6510  loss_bbox: 0.5445
    06/11 16:28:41 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][13/29]  lr: 1.0931e-03  eta: 0:05:28  time: 1.0960  data_time: 0.8874  memory: 2627  loss: 1.1959  loss_cls: 0.6500  loss_bbox: 0.5460
    06/11 16:28:42 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][14/29]  lr: 1.0971e-03  eta: 0:05:26  time: 1.0794  data_time: 0.8720  memory: 2603  loss: 1.1965  loss_cls: 0.6511  loss_bbox: 0.5454
    06/11 16:28:42 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][15/29]  lr: 1.1011e-03  eta: 0:05:24  time: 1.0794  data_time: 0.8720  memory: 2608  loss: 1.1978  loss_cls: 0.6501  loss_bbox: 0.5478
    06/11 16:28:42 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][16/29]  lr: 1.1051e-03  eta: 0:05:22  time: 1.0291  data_time: 0.8236  memory: 2616  loss: 1.1977  loss_cls: 0.6506  loss_bbox: 0.5470
    06/11 16:28:46 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][17/29]  lr: 1.1091e-03  eta: 0:05:24  time: 1.0983  data_time: 0.8935  memory: 2621  loss: 1.1965  loss_cls: 0.6505  loss_bbox: 0.5461
    06/11 16:28:46 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][18/29]  lr: 1.1131e-03  eta: 0:05:22  time: 1.0890  data_time: 0.8844  memory: 2613  loss: 1.1979  loss_cls: 0.6514  loss_bbox: 0.5465
    06/11 16:28:46 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][19/29]  lr: 1.1171e-03  eta: 0:05:20  time: 1.0880  data_time: 0.8841  memory: 2611  loss: 1.1987  loss_cls: 0.6518  loss_bbox: 0.5469
    06/11 16:28:46 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][20/29]  lr: 1.1211e-03  eta: 0:05:18  time: 1.0405  data_time: 0.8386  memory: 2608  loss: 1.1967  loss_cls: 0.6509  loss_bbox: 0.5458
    06/11 16:28:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][21/29]  lr: 1.1252e-03  eta: 0:05:18  time: 1.0900  data_time: 0.8865  memory: 2622  loss: 1.1909  loss_cls: 0.6497  loss_bbox: 0.5412
    06/11 16:28:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][22/29]  lr: 1.1292e-03  eta: 0:05:16  time: 1.0118  data_time: 0.8101  memory: 2609  loss: 1.1919  loss_cls: 0.6513  loss_bbox: 0.5407
    06/11 16:28:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][23/29]  lr: 1.1332e-03  eta: 0:05:14  time: 1.0087  data_time: 0.8083  memory: 2610  loss: 1.1909  loss_cls: 0.6499  loss_bbox: 0.5410
    06/11 16:28:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][24/29]  lr: 1.1372e-03  eta: 0:05:12  time: 1.0073  data_time: 0.8080  memory: 2626  loss: 1.1893  loss_cls: 0.6483  loss_bbox: 0.5409
    06/11 16:28:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][25/29]  lr: 1.1412e-03  eta: 0:05:14  time: 1.0649  data_time: 0.8634  memory: 2612  loss: 1.1871  loss_cls: 0.6465  loss_bbox: 0.5406
    06/11 16:28:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][26/29]  lr: 1.1452e-03  eta: 0:05:12  time: 1.0141  data_time: 0.8129  memory: 2616  loss: 1.1860  loss_cls: 0.6463  loss_bbox: 0.5398
    06/11 16:28:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][27/29]  lr: 1.1492e-03  eta: 0:05:10  time: 1.0067  data_time: 0.8064  memory: 2619  loss: 1.1861  loss_cls: 0.6465  loss_bbox: 0.5396
    06/11 16:28:54 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][28/29]  lr: 1.1532e-03  eta: 0:05:08  time: 1.0051  data_time: 0.8051  memory: 2613  loss: 1.1868  loss_cls: 0.6460  loss_bbox: 0.5408
    06/11 16:28:55 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:28:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [10][29/29]  lr: 1.1572e-03  eta: 0:05:07  time: 1.0194  data_time: 0.8193  memory: 1384  loss: 1.1870  loss_cls: 0.6451  loss_bbox: 0.5419
    06/11 16:28:55 - mmengine - [4m[97mINFO[0m - Saving checkpoint at 10 epochs
    06/11 16:29:01 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 1/14]    eta: 0:00:20  time: 0.6464  data_time: 0.5433  memory: 245  
    06/11 16:29:01 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 2/14]    eta: 0:00:10  time: 0.6284  data_time: 0.5257  memory: 245  
    06/11 16:29:02 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 3/14]    eta: 0:00:10  time: 0.6457  data_time: 0.5435  memory: 245  
    06/11 16:29:02 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 4/14]    eta: 0:00:07  time: 0.6278  data_time: 0.5266  memory: 245  
    06/11 16:29:03 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 5/14]    eta: 0:00:07  time: 0.6471  data_time: 0.5454  memory: 245  
    06/11 16:29:03 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 6/14]    eta: 0:00:05  time: 0.6310  data_time: 0.5295  memory: 245  
    06/11 16:29:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 7/14]    eta: 0:00:05  time: 0.6493  data_time: 0.5481  memory: 245  
    06/11 16:29:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 8/14]    eta: 0:00:04  time: 0.6331  data_time: 0.5329  memory: 245  
    06/11 16:29:06 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][ 9/14]    eta: 0:00:03  time: 0.6495  data_time: 0.5494  memory: 245  
    06/11 16:29:06 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][10/14]    eta: 0:00:02  time: 0.6344  data_time: 0.5350  memory: 245  
    06/11 16:29:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][11/14]    eta: 0:00:02  time: 0.6428  data_time: 0.5440  memory: 245  
    06/11 16:29:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][12/14]    eta: 0:00:01  time: 0.6342  data_time: 0.5344  memory: 245  
    06/11 16:29:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][13/14]    eta: 0:00:00  time: 0.6421  data_time: 0.5416  memory: 245  
    06/11 16:29:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][14/14]    eta: 0:00:00  time: 0.6369  data_time: 0.5370  memory: 245  
    06/11 16:29:09 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.12s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.48s).
    Accumulating evaluation results...
    DONE (t=0.15s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.235
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.402
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.277
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.235
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.652
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.654
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
    06/11 16:29:10 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.235 0.402 0.277 -1.000 -1.000 0.235
    06/11 16:29:10 - mmengine - [4m[97mINFO[0m - Epoch(val) [10][14/14]    coco/bbox_mAP: 0.2350  coco/bbox_mAP_50: 0.4020  coco/bbox_mAP_75: 0.2770  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.2350  data_time: 0.5605  time: 0.6572
    06/11 16:29:10 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_9.pth is removed
    06/11 16:29:13 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.2350 coco/bbox_mAP at 10 epoch is saved to best_coco_bbox_mAP_epoch_10.pth.
    06/11 16:29:16 - mmengine - [4m[97mINFO[0m - Switch pipeline now!
    06/11 16:29:19 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 1/29]  lr: 1.1612e-03  eta: 0:05:08  time: 1.0221  data_time: 0.8214  memory: 2602  loss: 1.1834  loss_cls: 0.6413  loss_bbox: 0.5421
    06/11 16:29:19 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 2/29]  lr: 1.1652e-03  eta: 0:05:06  time: 1.0220  data_time: 0.8212  memory: 2602  loss: 1.1847  loss_cls: 0.6428  loss_bbox: 0.5419
    06/11 16:29:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 3/29]  lr: 1.1692e-03  eta: 0:05:04  time: 1.0128  data_time: 0.8116  memory: 2602  loss: 1.1874  loss_cls: 0.6441  loss_bbox: 0.5433
    06/11 16:29:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 4/29]  lr: 1.1732e-03  eta: 0:05:02  time: 1.0058  data_time: 0.8039  memory: 2602  loss: 1.1893  loss_cls: 0.6451  loss_bbox: 0.5442
    06/11 16:29:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 5/29]  lr: 1.1772e-03  eta: 0:05:01  time: 0.9915  data_time: 0.7909  memory: 2602  loss: 1.1916  loss_cls: 0.6439  loss_bbox: 0.5477
    06/11 16:29:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 6/29]  lr: 1.1812e-03  eta: 0:05:00  time: 0.9903  data_time: 0.7904  memory: 2602  loss: 1.1934  loss_cls: 0.6437  loss_bbox: 0.5497
    06/11 16:29:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 7/29]  lr: 1.1852e-03  eta: 0:04:58  time: 0.9723  data_time: 0.7717  memory: 2602  loss: 1.1936  loss_cls: 0.6455  loss_bbox: 0.5482
    06/11 16:29:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 8/29]  lr: 1.1892e-03  eta: 0:04:56  time: 0.9693  data_time: 0.7692  memory: 2602  loss: 1.1932  loss_cls: 0.6448  loss_bbox: 0.5484
    06/11 16:29:24 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][ 9/29]  lr: 1.1932e-03  eta: 0:04:56  time: 0.9802  data_time: 0.7797  memory: 2602  loss: 1.1927  loss_cls: 0.6457  loss_bbox: 0.5470
    06/11 16:29:24 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][10/29]  lr: 1.1972e-03  eta: 0:04:54  time: 0.9765  data_time: 0.7767  memory: 2602  loss: 1.1916  loss_cls: 0.6454  loss_bbox: 0.5462
    06/11 16:29:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][11/29]  lr: 1.2012e-03  eta: 0:04:52  time: 0.9560  data_time: 0.7552  memory: 2602  loss: 1.1904  loss_cls: 0.6449  loss_bbox: 0.5455
    06/11 16:29:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][12/29]  lr: 1.2052e-03  eta: 0:04:51  time: 0.9413  data_time: 0.7406  memory: 2602  loss: 1.1889  loss_cls: 0.6439  loss_bbox: 0.5450
    06/11 16:29:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][13/29]  lr: 1.2092e-03  eta: 0:04:50  time: 0.9566  data_time: 0.7561  memory: 2602  loss: 1.1856  loss_cls: 0.6434  loss_bbox: 0.5421
    06/11 16:29:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][14/29]  lr: 1.2132e-03  eta: 0:04:48  time: 0.9435  data_time: 0.7451  memory: 2602  loss: 1.1833  loss_cls: 0.6416  loss_bbox: 0.5417
    06/11 16:29:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][15/29]  lr: 1.2172e-03  eta: 0:04:47  time: 0.9288  data_time: 0.7330  memory: 2602  loss: 1.1844  loss_cls: 0.6420  loss_bbox: 0.5425
    06/11 16:29:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][16/29]  lr: 1.2212e-03  eta: 0:04:45  time: 0.9037  data_time: 0.7097  memory: 2602  loss: 1.1829  loss_cls: 0.6419  loss_bbox: 0.5410
    06/11 16:29:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][17/29]  lr: 1.2253e-03  eta: 0:04:44  time: 0.9251  data_time: 0.7331  memory: 2602  loss: 1.1841  loss_cls: 0.6451  loss_bbox: 0.5389
    06/11 16:29:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][18/29]  lr: 1.2293e-03  eta: 0:04:42  time: 0.9138  data_time: 0.7215  memory: 2602  loss: 1.1827  loss_cls: 0.6442  loss_bbox: 0.5384
    06/11 16:29:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][19/29]  lr: 1.2333e-03  eta: 0:04:41  time: 0.8977  data_time: 0.7074  memory: 2602  loss: 1.1831  loss_cls: 0.6450  loss_bbox: 0.5382
    06/11 16:29:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][20/29]  lr: 1.2373e-03  eta: 0:04:39  time: 0.8821  data_time: 0.6926  memory: 2602  loss: 1.1793  loss_cls: 0.6430  loss_bbox: 0.5363
    06/11 16:29:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][21/29]  lr: 1.2413e-03  eta: 0:04:38  time: 0.9139  data_time: 0.7227  memory: 2602  loss: 1.1735  loss_cls: 0.6392  loss_bbox: 0.5343
    06/11 16:29:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][22/29]  lr: 1.2453e-03  eta: 0:04:37  time: 0.8306  data_time: 0.6411  memory: 2602  loss: 1.1700  loss_cls: 0.6380  loss_bbox: 0.5321
    06/11 16:29:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][23/29]  lr: 1.2493e-03  eta: 0:04:35  time: 0.8318  data_time: 0.6426  memory: 2602  loss: 1.1656  loss_cls: 0.6349  loss_bbox: 0.5307
    06/11 16:29:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][24/29]  lr: 1.2533e-03  eta: 0:04:33  time: 0.8307  data_time: 0.6419  memory: 2602  loss: 1.1623  loss_cls: 0.6325  loss_bbox: 0.5298
    06/11 16:29:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][25/29]  lr: 1.2573e-03  eta: 0:04:33  time: 0.8560  data_time: 0.6676  memory: 2602  loss: 1.1564  loss_cls: 0.6306  loss_bbox: 0.5258
    06/11 16:29:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][26/29]  lr: 1.2613e-03  eta: 0:04:31  time: 0.7910  data_time: 0.6041  memory: 2602  loss: 1.1572  loss_cls: 0.6321  loss_bbox: 0.5251
    06/11 16:29:35 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][27/29]  lr: 1.2653e-03  eta: 0:04:30  time: 0.7949  data_time: 0.6084  memory: 2602  loss: 1.1569  loss_cls: 0.6334  loss_bbox: 0.5236
    06/11 16:29:35 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][28/29]  lr: 1.2693e-03  eta: 0:04:28  time: 0.7953  data_time: 0.6092  memory: 2602  loss: 1.1536  loss_cls: 0.6325  loss_bbox: 0.5210
    06/11 16:29:35 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:29:35 - mmengine - [4m[97mINFO[0m - Epoch(train) [11][29/29]  lr: 1.2733e-03  eta: 0:04:27  time: 0.8007  data_time: 0.6150  memory: 1345  loss: 1.1514  loss_cls: 0.6319  loss_bbox: 0.5195
    06/11 16:29:37 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 1/14]    eta: 0:00:19  time: 0.6565  data_time: 0.5559  memory: 245  
    06/11 16:29:37 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 2/14]    eta: 0:00:09  time: 0.6442  data_time: 0.5438  memory: 245  
    06/11 16:29:38 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 3/14]    eta: 0:00:09  time: 0.6519  data_time: 0.5515  memory: 245  
    06/11 16:29:38 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 4/14]    eta: 0:00:07  time: 0.6432  data_time: 0.5423  memory: 245  
    06/11 16:29:39 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 5/14]    eta: 0:00:06  time: 0.6519  data_time: 0.5508  memory: 245  
    06/11 16:29:40 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 6/14]    eta: 0:00:05  time: 0.6411  data_time: 0.5395  memory: 245  
    06/11 16:29:41 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 7/14]    eta: 0:00:05  time: 0.6542  data_time: 0.5523  memory: 245  
    06/11 16:29:41 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 8/14]    eta: 0:00:04  time: 0.6443  data_time: 0.5419  memory: 245  
    06/11 16:29:42 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][ 9/14]    eta: 0:00:03  time: 0.6123  data_time: 0.5104  memory: 245  
    06/11 16:29:42 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][10/14]    eta: 0:00:02  time: 0.6115  data_time: 0.5098  memory: 245  
    06/11 16:29:43 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][11/14]    eta: 0:00:02  time: 0.6110  data_time: 0.5100  memory: 245  
    06/11 16:29:44 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][12/14]    eta: 0:00:01  time: 0.6140  data_time: 0.5124  memory: 245  
    06/11 16:29:45 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][13/14]    eta: 0:00:00  time: 0.6100  data_time: 0.5079  memory: 245  
    06/11 16:29:45 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][14/14]    eta: 0:00:00  time: 0.6176  data_time: 0.5150  memory: 245  
    06/11 16:29:45 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.50s).
    Accumulating evaluation results...
    DONE (t=0.15s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.210
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.478
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.122
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.210
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.277
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.571
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.594
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.594
    06/11 16:29:46 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.210 0.478 0.122 -1.000 -1.000 0.210
    06/11 16:29:46 - mmengine - [4m[97mINFO[0m - Epoch(val) [11][14/14]    coco/bbox_mAP: 0.2100  coco/bbox_mAP_50: 0.4780  coco/bbox_mAP_75: 0.1220  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.2100  data_time: 0.5449  time: 0.6562
    06/11 16:29:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 1/29]  lr: 1.2773e-03  eta: 0:04:27  time: 0.7874  data_time: 0.6022  memory: 2602  loss: 1.1491  loss_cls: 0.6305  loss_bbox: 0.5186
    06/11 16:29:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 2/29]  lr: 1.2813e-03  eta: 0:04:25  time: 0.7903  data_time: 0.6038  memory: 2602  loss: 1.1484  loss_cls: 0.6266  loss_bbox: 0.5218
    06/11 16:29:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 3/29]  lr: 1.2853e-03  eta: 0:04:24  time: 0.7902  data_time: 0.6037  memory: 2602  loss: 1.1465  loss_cls: 0.6275  loss_bbox: 0.5190
    06/11 16:29:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 4/29]  lr: 1.2893e-03  eta: 0:04:22  time: 0.7901  data_time: 0.6040  memory: 2602  loss: 1.1461  loss_cls: 0.6281  loss_bbox: 0.5180
    06/11 16:29:51 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 5/29]  lr: 1.2933e-03  eta: 0:04:21  time: 0.7385  data_time: 0.5519  memory: 2602  loss: 1.1421  loss_cls: 0.6271  loss_bbox: 0.5149
    06/11 16:29:51 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 6/29]  lr: 1.2973e-03  eta: 0:04:20  time: 0.7449  data_time: 0.5579  memory: 2602  loss: 1.1403  loss_cls: 0.6258  loss_bbox: 0.5145
    06/11 16:29:52 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 7/29]  lr: 1.3013e-03  eta: 0:04:18  time: 0.7446  data_time: 0.5579  memory: 2602  loss: 1.1425  loss_cls: 0.6291  loss_bbox: 0.5134
    06/11 16:29:52 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 8/29]  lr: 1.3053e-03  eta: 0:04:16  time: 0.7437  data_time: 0.5574  memory: 2602  loss: 1.1369  loss_cls: 0.6269  loss_bbox: 0.5100
    06/11 16:29:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][ 9/29]  lr: 1.3093e-03  eta: 0:04:15  time: 0.6931  data_time: 0.5061  memory: 2602  loss: 1.1369  loss_cls: 0.6282  loss_bbox: 0.5087
    06/11 16:29:54 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][10/29]  lr: 1.3133e-03  eta: 0:04:14  time: 0.7099  data_time: 0.5238  memory: 2602  loss: 1.1321  loss_cls: 0.6261  loss_bbox: 0.5059
    06/11 16:29:54 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][11/29]  lr: 1.3173e-03  eta: 0:04:13  time: 0.7096  data_time: 0.5237  memory: 2602  loss: 1.1312  loss_cls: 0.6267  loss_bbox: 0.5045
    06/11 16:29:54 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][12/29]  lr: 1.3213e-03  eta: 0:04:11  time: 0.7104  data_time: 0.5246  memory: 2602  loss: 1.1299  loss_cls: 0.6272  loss_bbox: 0.5027
    06/11 16:29:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][13/29]  lr: 1.3254e-03  eta: 0:04:10  time: 0.6730  data_time: 0.4876  memory: 2602  loss: 1.1282  loss_cls: 0.6265  loss_bbox: 0.5017
    06/11 16:29:56 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][14/29]  lr: 1.3294e-03  eta: 0:04:09  time: 0.6898  data_time: 0.5044  memory: 2602  loss: 1.1231  loss_cls: 0.6241  loss_bbox: 0.4990
    06/11 16:29:56 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][15/29]  lr: 1.3334e-03  eta: 0:04:07  time: 0.6899  data_time: 0.5046  memory: 2602  loss: 1.1173  loss_cls: 0.6217  loss_bbox: 0.4956
    06/11 16:29:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][16/29]  lr: 1.3374e-03  eta: 0:04:06  time: 0.6902  data_time: 0.5051  memory: 2602  loss: 1.1129  loss_cls: 0.6204  loss_bbox: 0.4925
    06/11 16:29:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][17/29]  lr: 1.3414e-03  eta: 0:04:05  time: 0.6447  data_time: 0.4626  memory: 2602  loss: 1.1089  loss_cls: 0.6193  loss_bbox: 0.4896
    06/11 16:29:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][18/29]  lr: 1.3454e-03  eta: 0:04:04  time: 0.6557  data_time: 0.4733  memory: 2602  loss: 1.1055  loss_cls: 0.6178  loss_bbox: 0.4877
    06/11 16:29:59 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][19/29]  lr: 1.3494e-03  eta: 0:04:02  time: 0.6588  data_time: 0.4747  memory: 2602  loss: 1.1019  loss_cls: 0.6142  loss_bbox: 0.4876
    06/11 16:29:59 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][20/29]  lr: 1.3534e-03  eta: 0:04:01  time: 0.6599  data_time: 0.4742  memory: 2602  loss: 1.0968  loss_cls: 0.6124  loss_bbox: 0.4844
    06/11 16:30:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][21/29]  lr: 1.3574e-03  eta: 0:04:00  time: 0.6570  data_time: 0.4705  memory: 2602  loss: 1.0913  loss_cls: 0.6108  loss_bbox: 0.4805
    06/11 16:30:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][22/29]  lr: 1.3614e-03  eta: 0:03:58  time: 0.6032  data_time: 0.4187  memory: 2602  loss: 1.0881  loss_cls: 0.6104  loss_bbox: 0.4777
    06/11 16:30:01 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][23/29]  lr: 1.3654e-03  eta: 0:03:57  time: 0.6053  data_time: 0.4208  memory: 2602  loss: 1.0821  loss_cls: 0.6073  loss_bbox: 0.4748
    06/11 16:30:01 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][24/29]  lr: 1.3694e-03  eta: 0:03:55  time: 0.6041  data_time: 0.4205  memory: 2602  loss: 1.0796  loss_cls: 0.6055  loss_bbox: 0.4741
    06/11 16:30:03 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][25/29]  lr: 1.3734e-03  eta: 0:03:55  time: 0.6357  data_time: 0.4514  memory: 2602  loss: 1.0734  loss_cls: 0.6022  loss_bbox: 0.4712
    06/11 16:30:03 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][26/29]  lr: 1.3774e-03  eta: 0:03:53  time: 0.6091  data_time: 0.4247  memory: 2602  loss: 1.0669  loss_cls: 0.6007  loss_bbox: 0.4662
    06/11 16:30:03 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][27/29]  lr: 1.3814e-03  eta: 0:03:52  time: 0.6109  data_time: 0.4264  memory: 2602  loss: 1.0613  loss_cls: 0.5987  loss_bbox: 0.4626
    06/11 16:30:03 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][28/29]  lr: 1.3854e-03  eta: 0:03:50  time: 0.6028  data_time: 0.4196  memory: 2602  loss: 1.0565  loss_cls: 0.5951  loss_bbox: 0.4614
    06/11 16:30:04 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:30:04 - mmengine - [4m[97mINFO[0m - Epoch(train) [12][29/29]  lr: 1.3894e-03  eta: 0:03:49  time: 0.6089  data_time: 0.4255  memory: 1345  loss: 1.0532  loss_cls: 0.5939  loss_bbox: 0.4594
    06/11 16:30:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 1/14]    eta: 0:00:18  time: 0.6202  data_time: 0.5176  memory: 245  
    06/11 16:30:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 2/14]    eta: 0:00:09  time: 0.6202  data_time: 0.5176  memory: 245  
    06/11 16:30:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 3/14]    eta: 0:00:10  time: 0.6245  data_time: 0.5209  memory: 245  
    06/11 16:30:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 4/14]    eta: 0:00:07  time: 0.6220  data_time: 0.5181  memory: 245  
    06/11 16:30:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 5/14]    eta: 0:00:07  time: 0.6273  data_time: 0.5237  memory: 245  
    06/11 16:30:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 6/14]    eta: 0:00:05  time: 0.6158  data_time: 0.5122  memory: 245  
    06/11 16:30:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 7/14]    eta: 0:00:05  time: 0.6328  data_time: 0.5292  memory: 245  
    06/11 16:30:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 8/14]    eta: 0:00:04  time: 0.6165  data_time: 0.5140  memory: 245  
    06/11 16:30:11 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][ 9/14]    eta: 0:00:03  time: 0.6190  data_time: 0.5153  memory: 245  
    06/11 16:30:11 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][10/14]    eta: 0:00:02  time: 0.6196  data_time: 0.5152  memory: 245  
    06/11 16:30:12 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][11/14]    eta: 0:00:02  time: 0.6242  data_time: 0.5189  memory: 245  
    06/11 16:30:12 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][12/14]    eta: 0:00:01  time: 0.6248  data_time: 0.5189  memory: 245  
    06/11 16:30:13 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][13/14]    eta: 0:00:00  time: 0.6299  data_time: 0.5239  memory: 245  
    06/11 16:30:13 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][14/14]    eta: 0:00:00  time: 0.6301  data_time: 0.5240  memory: 245  
    06/11 16:30:13 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.51s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.723
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.437
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.431
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.503
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.673
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.675
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.675
    06/11 16:30:14 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.431 0.723 0.437 -1.000 -1.000 0.431
    06/11 16:30:14 - mmengine - [4m[97mINFO[0m - Epoch(val) [12][14/14]    coco/bbox_mAP: 0.4310  coco/bbox_mAP_50: 0.7230  coco/bbox_mAP_75: 0.4370  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.4310  data_time: 0.5251  time: 0.6446
    06/11 16:30:14 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_10.pth is removed
    06/11 16:30:18 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.4310 coco/bbox_mAP at 12 epoch is saved to best_coco_bbox_mAP_epoch_12.pth.
    06/11 16:30:24 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 1/29]  lr: 1.3934e-03  eta: 0:03:49  time: 0.6289  data_time: 0.4432  memory: 2602  loss: 1.0499  loss_cls: 0.5904  loss_bbox: 0.4595
    06/11 16:30:24 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 2/29]  lr: 1.3974e-03  eta: 0:03:47  time: 0.6293  data_time: 0.4430  memory: 2602  loss: 1.0436  loss_cls: 0.5865  loss_bbox: 0.4571
    06/11 16:30:24 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 3/29]  lr: 1.4014e-03  eta: 0:03:46  time: 0.6229  data_time: 0.4375  memory: 2602  loss: 1.0399  loss_cls: 0.5853  loss_bbox: 0.4547
    06/11 16:30:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 4/29]  lr: 1.4054e-03  eta: 0:03:44  time: 0.6207  data_time: 0.4369  memory: 2602  loss: 1.0362  loss_cls: 0.5824  loss_bbox: 0.4538
    06/11 16:30:26 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 5/29]  lr: 1.4094e-03  eta: 0:03:44  time: 0.6236  data_time: 0.4416  memory: 2602  loss: 1.0317  loss_cls: 0.5796  loss_bbox: 0.4521
    06/11 16:30:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 6/29]  lr: 1.4134e-03  eta: 0:03:43  time: 0.6236  data_time: 0.4419  memory: 2602  loss: 1.0264  loss_cls: 0.5773  loss_bbox: 0.4491
    06/11 16:30:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 7/29]  lr: 1.4174e-03  eta: 0:03:41  time: 0.6245  data_time: 0.4426  memory: 2602  loss: 1.0217  loss_cls: 0.5751  loss_bbox: 0.4466
    06/11 16:30:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 8/29]  lr: 1.4214e-03  eta: 0:03:40  time: 0.6244  data_time: 0.4429  memory: 2602  loss: 1.0178  loss_cls: 0.5710  loss_bbox: 0.4468
    06/11 16:30:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][ 9/29]  lr: 1.4255e-03  eta: 0:03:39  time: 0.6255  data_time: 0.4434  memory: 2602  loss: 1.0131  loss_cls: 0.5679  loss_bbox: 0.4452
    06/11 16:30:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][10/29]  lr: 1.4295e-03  eta: 0:03:38  time: 0.6247  data_time: 0.4435  memory: 2602  loss: 1.0174  loss_cls: 0.5749  loss_bbox: 0.4425
    06/11 16:30:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][11/29]  lr: 1.4335e-03  eta: 0:03:36  time: 0.6272  data_time: 0.4443  memory: 2602  loss: 1.0132  loss_cls: 0.5728  loss_bbox: 0.4403
    06/11 16:30:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][12/29]  lr: 1.4375e-03  eta: 0:03:35  time: 0.6282  data_time: 0.4442  memory: 2602  loss: 1.0109  loss_cls: 0.5716  loss_bbox: 0.4393
    06/11 16:30:31 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][13/29]  lr: 1.4415e-03  eta: 0:03:34  time: 0.6289  data_time: 0.4435  memory: 2602  loss: 1.0102  loss_cls: 0.5714  loss_bbox: 0.4388
    06/11 16:30:31 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][14/29]  lr: 1.4455e-03  eta: 0:03:33  time: 0.6292  data_time: 0.4431  memory: 2602  loss: 1.0064  loss_cls: 0.5703  loss_bbox: 0.4361
    06/11 16:30:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][15/29]  lr: 1.4495e-03  eta: 0:03:31  time: 0.6295  data_time: 0.4424  memory: 2602  loss: 1.0072  loss_cls: 0.5719  loss_bbox: 0.4353
    06/11 16:30:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][16/29]  lr: 1.4535e-03  eta: 0:03:30  time: 0.6302  data_time: 0.4424  memory: 2604  loss: 1.0044  loss_cls: 0.5697  loss_bbox: 0.4346
    06/11 16:30:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][17/29]  lr: 1.4575e-03  eta: 0:03:29  time: 0.6277  data_time: 0.4375  memory: 2602  loss: 1.0046  loss_cls: 0.5709  loss_bbox: 0.4337
    06/11 16:30:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][18/29]  lr: 1.4615e-03  eta: 0:03:28  time: 0.6285  data_time: 0.4373  memory: 2602  loss: 0.9976  loss_cls: 0.5679  loss_bbox: 0.4297
    06/11 16:30:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][19/29]  lr: 1.4655e-03  eta: 0:03:27  time: 0.6229  data_time: 0.4321  memory: 2602  loss: 0.9956  loss_cls: 0.5662  loss_bbox: 0.4294
    06/11 16:30:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][20/29]  lr: 1.4695e-03  eta: 0:03:25  time: 0.6226  data_time: 0.4318  memory: 2605  loss: 0.9929  loss_cls: 0.5643  loss_bbox: 0.4286
    06/11 16:30:36 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][21/29]  lr: 1.4735e-03  eta: 0:03:25  time: 0.6427  data_time: 0.4487  memory: 2605  loss: 0.9912  loss_cls: 0.5625  loss_bbox: 0.4287
    06/11 16:30:36 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][22/29]  lr: 1.4775e-03  eta: 0:03:23  time: 0.5976  data_time: 0.4022  memory: 2602  loss: 0.9874  loss_cls: 0.5611  loss_bbox: 0.4263
    06/11 16:30:36 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][23/29]  lr: 1.4815e-03  eta: 0:03:22  time: 0.5964  data_time: 0.4012  memory: 2602  loss: 0.9792  loss_cls: 0.5588  loss_bbox: 0.4203
    06/11 16:30:37 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][24/29]  lr: 1.4855e-03  eta: 0:03:20  time: 0.5958  data_time: 0.4011  memory: 2602  loss: 0.9753  loss_cls: 0.5556  loss_bbox: 0.4198
    06/11 16:30:38 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][25/29]  lr: 1.4895e-03  eta: 0:03:20  time: 0.6182  data_time: 0.4216  memory: 2602  loss: 0.9685  loss_cls: 0.5524  loss_bbox: 0.4161
    06/11 16:30:38 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][26/29]  lr: 1.4935e-03  eta: 0:03:18  time: 0.5998  data_time: 0.4044  memory: 2602  loss: 0.9653  loss_cls: 0.5506  loss_bbox: 0.4147
    06/11 16:30:38 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][27/29]  lr: 1.4975e-03  eta: 0:03:17  time: 0.5932  data_time: 0.3985  memory: 2602  loss: 0.9629  loss_cls: 0.5494  loss_bbox: 0.4135
    06/11 16:30:39 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][28/29]  lr: 1.5015e-03  eta: 0:03:16  time: 0.5935  data_time: 0.3988  memory: 2602  loss: 0.9529  loss_cls: 0.5433  loss_bbox: 0.4097
    06/11 16:30:39 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:30:39 - mmengine - [4m[97mINFO[0m - Epoch(train) [13][29/29]  lr: 1.5055e-03  eta: 0:03:14  time: 0.5998  data_time: 0.4045  memory: 1346  loss: 0.9518  loss_cls: 0.5423  loss_bbox: 0.4094
    06/11 16:30:40 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 1/14]    eta: 0:00:17  time: 0.6401  data_time: 0.5340  memory: 245  
    06/11 16:30:41 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 2/14]    eta: 0:00:09  time: 0.6438  data_time: 0.5370  memory: 245  
    06/11 16:30:42 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 3/14]    eta: 0:00:09  time: 0.6488  data_time: 0.5422  memory: 245  
    06/11 16:30:42 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 4/14]    eta: 0:00:07  time: 0.6524  data_time: 0.5454  memory: 245  
    06/11 16:30:43 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 5/14]    eta: 0:00:06  time: 0.6613  data_time: 0.5544  memory: 245  
    06/11 16:30:43 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 6/14]    eta: 0:00:05  time: 0.6629  data_time: 0.5550  memory: 245  
    06/11 16:30:44 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 7/14]    eta: 0:00:05  time: 0.6763  data_time: 0.5674  memory: 245  
    06/11 16:30:45 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 8/14]    eta: 0:00:04  time: 0.6726  data_time: 0.5642  memory: 245  
    06/11 16:30:46 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][ 9/14]    eta: 0:00:03  time: 0.6661  data_time: 0.5573  memory: 245  
    06/11 16:30:46 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][10/14]    eta: 0:00:02  time: 0.6660  data_time: 0.5570  memory: 245  
    06/11 16:30:47 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][11/14]    eta: 0:00:02  time: 0.6622  data_time: 0.5529  memory: 245  
    06/11 16:30:47 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][12/14]    eta: 0:00:01  time: 0.6664  data_time: 0.5556  memory: 245  
    06/11 16:30:48 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][13/14]    eta: 0:00:00  time: 0.6595  data_time: 0.5483  memory: 245  
    06/11 16:30:48 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][14/14]    eta: 0:00:00  time: 0.6636  data_time: 0.5526  memory: 245  
    06/11 16:30:49 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.18s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.58s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.684
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.486
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.424
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.488
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.707
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.708
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
    06/11 16:30:50 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.424 0.684 0.486 -1.000 -1.000 0.424
    06/11 16:30:50 - mmengine - [4m[97mINFO[0m - Epoch(val) [13][14/14]    coco/bbox_mAP: 0.4240  coco/bbox_mAP_50: 0.6840  coco/bbox_mAP_75: 0.4860  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.4240  data_time: 0.5160  time: 0.6254
    06/11 16:30:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 1/29]  lr: 1.5095e-03  eta: 0:03:14  time: 0.6355  data_time: 0.4386  memory: 2605  loss: 0.9482  loss_cls: 0.5391  loss_bbox: 0.4091
    06/11 16:30:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 2/29]  lr: 1.5135e-03  eta: 0:03:13  time: 0.6182  data_time: 0.4215  memory: 2602  loss: 0.9459  loss_cls: 0.5385  loss_bbox: 0.4074
    06/11 16:30:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 3/29]  lr: 1.5175e-03  eta: 0:03:12  time: 0.6193  data_time: 0.4222  memory: 2602  loss: 0.9408  loss_cls: 0.5345  loss_bbox: 0.4063
    06/11 16:30:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 4/29]  lr: 1.5215e-03  eta: 0:03:10  time: 0.6220  data_time: 0.4242  memory: 2602  loss: 0.9366  loss_cls: 0.5317  loss_bbox: 0.4049
    06/11 16:30:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 5/29]  lr: 1.5256e-03  eta: 0:03:10  time: 0.6343  data_time: 0.4368  memory: 2602  loss: 0.9338  loss_cls: 0.5299  loss_bbox: 0.4039
    06/11 16:30:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 6/29]  lr: 1.5296e-03  eta: 0:03:08  time: 0.6175  data_time: 0.4201  memory: 2602  loss: 0.9272  loss_cls: 0.5259  loss_bbox: 0.4013
    06/11 16:30:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 7/29]  lr: 1.5336e-03  eta: 0:03:07  time: 0.6175  data_time: 0.4201  memory: 2605  loss: 0.9251  loss_cls: 0.5233  loss_bbox: 0.4018
    06/11 16:30:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 8/29]  lr: 1.5376e-03  eta: 0:03:06  time: 0.6169  data_time: 0.4195  memory: 2602  loss: 0.9274  loss_cls: 0.5236  loss_bbox: 0.4038
    06/11 16:30:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][ 9/29]  lr: 1.5416e-03  eta: 0:03:05  time: 0.6343  data_time: 0.4371  memory: 2602  loss: 0.9275  loss_cls: 0.5244  loss_bbox: 0.4031
    06/11 16:30:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][10/29]  lr: 1.5456e-03  eta: 0:03:04  time: 0.6230  data_time: 0.4268  memory: 2602  loss: 0.9223  loss_cls: 0.5206  loss_bbox: 0.4017
    06/11 16:30:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][11/29]  lr: 1.5496e-03  eta: 0:03:03  time: 0.6202  data_time: 0.4258  memory: 2602  loss: 0.9213  loss_cls: 0.5197  loss_bbox: 0.4016
    06/11 16:30:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][12/29]  lr: 1.5536e-03  eta: 0:03:01  time: 0.6182  data_time: 0.4257  memory: 2602  loss: 0.9184  loss_cls: 0.5177  loss_bbox: 0.4007
    06/11 16:31:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][13/29]  lr: 1.5576e-03  eta: 0:03:01  time: 0.6311  data_time: 0.4382  memory: 2602  loss: 0.9170  loss_cls: 0.5159  loss_bbox: 0.4011
    06/11 16:31:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][14/29]  lr: 1.5616e-03  eta: 0:02:59  time: 0.6301  data_time: 0.4372  memory: 2602  loss: 0.9129  loss_cls: 0.5114  loss_bbox: 0.4015
    06/11 16:31:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][15/29]  lr: 1.5656e-03  eta: 0:02:58  time: 0.6287  data_time: 0.4359  memory: 2602  loss: 0.9096  loss_cls: 0.5091  loss_bbox: 0.4005
    06/11 16:31:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][16/29]  lr: 1.5696e-03  eta: 0:02:57  time: 0.6287  data_time: 0.4357  memory: 2602  loss: 0.9021  loss_cls: 0.5046  loss_bbox: 0.3975
    06/11 16:31:02 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][17/29]  lr: 1.5736e-03  eta: 0:02:56  time: 0.6296  data_time: 0.4380  memory: 2602  loss: 0.9006  loss_cls: 0.5032  loss_bbox: 0.3974
    06/11 16:31:02 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][18/29]  lr: 1.5776e-03  eta: 0:02:55  time: 0.6290  data_time: 0.4376  memory: 2602  loss: 0.8961  loss_cls: 0.4999  loss_bbox: 0.3962
    06/11 16:31:03 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][19/29]  lr: 1.5816e-03  eta: 0:02:54  time: 0.6281  data_time: 0.4366  memory: 2602  loss: 0.8914  loss_cls: 0.4979  loss_bbox: 0.3935
    06/11 16:31:03 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][20/29]  lr: 1.5856e-03  eta: 0:02:52  time: 0.6293  data_time: 0.4371  memory: 2602  loss: 0.8863  loss_cls: 0.4947  loss_bbox: 0.3916
    06/11 16:31:04 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][21/29]  lr: 1.5896e-03  eta: 0:02:52  time: 0.6485  data_time: 0.4558  memory: 2602  loss: 0.8819  loss_cls: 0.4907  loss_bbox: 0.3912
    06/11 16:31:04 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][22/29]  lr: 1.5936e-03  eta: 0:02:50  time: 0.5981  data_time: 0.4079  memory: 2602  loss: 0.8797  loss_cls: 0.4907  loss_bbox: 0.3890
    06/11 16:31:05 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][23/29]  lr: 1.5976e-03  eta: 0:02:49  time: 0.5975  data_time: 0.4079  memory: 2602  loss: 0.8815  loss_cls: 0.4912  loss_bbox: 0.3903
    06/11 16:31:05 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][24/29]  lr: 1.6016e-03  eta: 0:02:48  time: 0.5966  data_time: 0.4075  memory: 2602  loss: 0.8796  loss_cls: 0.4894  loss_bbox: 0.3902
    06/11 16:31:06 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][25/29]  lr: 1.6056e-03  eta: 0:02:47  time: 0.6273  data_time: 0.4369  memory: 2602  loss: 0.8787  loss_cls: 0.4884  loss_bbox: 0.3903
    06/11 16:31:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][26/29]  lr: 1.6096e-03  eta: 0:02:46  time: 0.5950  data_time: 0.4046  memory: 2602  loss: 0.8767  loss_cls: 0.4869  loss_bbox: 0.3898
    06/11 16:31:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][27/29]  lr: 1.6136e-03  eta: 0:02:45  time: 0.5948  data_time: 0.4043  memory: 2602  loss: 0.8761  loss_cls: 0.4860  loss_bbox: 0.3901
    06/11 16:31:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][28/29]  lr: 1.6176e-03  eta: 0:02:43  time: 0.5945  data_time: 0.4041  memory: 2602  loss: 0.8757  loss_cls: 0.4842  loss_bbox: 0.3915
    06/11 16:31:07 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:31:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [14][29/29]  lr: 1.6216e-03  eta: 0:02:42  time: 0.5963  data_time: 0.4059  memory: 1345  loss: 0.8753  loss_cls: 0.4856  loss_bbox: 0.3897
    06/11 16:31:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 1/14]    eta: 0:00:18  time: 0.6673  data_time: 0.5554  memory: 245  
    06/11 16:31:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 2/14]    eta: 0:00:09  time: 0.6691  data_time: 0.5555  memory: 245  
    06/11 16:31:10 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 3/14]    eta: 0:00:10  time: 0.6669  data_time: 0.5522  memory: 245  
    06/11 16:31:11 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 4/14]    eta: 0:00:07  time: 0.6680  data_time: 0.5523  memory: 245  
    06/11 16:31:12 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 5/14]    eta: 0:00:07  time: 0.6720  data_time: 0.5549  memory: 245  
    06/11 16:31:12 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 6/14]    eta: 0:00:05  time: 0.6683  data_time: 0.5517  memory: 245  
    06/11 16:31:13 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 7/14]    eta: 0:00:05  time: 0.6699  data_time: 0.5529  memory: 245  
    06/11 16:31:13 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 8/14]    eta: 0:00:04  time: 0.6660  data_time: 0.5478  memory: 245  
    06/11 16:31:14 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][ 9/14]    eta: 0:00:03  time: 0.6569  data_time: 0.5385  memory: 245  
    06/11 16:31:14 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][10/14]    eta: 0:00:02  time: 0.6564  data_time: 0.5381  memory: 245  
    06/11 16:31:15 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][11/14]    eta: 0:00:02  time: 0.6573  data_time: 0.5388  memory: 245  
    06/11 16:31:16 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][12/14]    eta: 0:00:01  time: 0.6562  data_time: 0.5377  memory: 245  
    06/11 16:31:16 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][13/14]    eta: 0:00:00  time: 0.6539  data_time: 0.5357  memory: 245  
    06/11 16:31:17 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][14/14]    eta: 0:00:00  time: 0.6592  data_time: 0.5419  memory: 245  
    06/11 16:31:17 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.28s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.62s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.829
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.470
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.470
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.497
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.650
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.654
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
    06/11 16:31:18 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.470 0.829 0.470 -1.000 -1.000 0.470
    06/11 16:31:18 - mmengine - [4m[97mINFO[0m - Epoch(val) [14][14/14]    coco/bbox_mAP: 0.4700  coco/bbox_mAP_50: 0.8290  coco/bbox_mAP_75: 0.4700  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.4700  data_time: 0.5051  time: 0.6269
    06/11 16:31:18 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_12.pth is removed
    06/11 16:31:22 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.4700 coco/bbox_mAP at 14 epoch is saved to best_coco_bbox_mAP_epoch_14.pth.
    06/11 16:31:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 1/29]  lr: 1.6256e-03  eta: 0:02:42  time: 0.6112  data_time: 0.4212  memory: 2602  loss: 0.8730  loss_cls: 0.4828  loss_bbox: 0.3902
    06/11 16:31:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 2/29]  lr: 1.6297e-03  eta: 0:02:41  time: 0.6118  data_time: 0.4214  memory: 2602  loss: 0.8618  loss_cls: 0.4722  loss_bbox: 0.3897
    06/11 16:31:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 3/29]  lr: 1.6337e-03  eta: 0:02:39  time: 0.6099  data_time: 0.4209  memory: 2602  loss: 0.8619  loss_cls: 0.4728  loss_bbox: 0.3891
    06/11 16:31:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 4/29]  lr: 1.6377e-03  eta: 0:02:38  time: 0.6091  data_time: 0.4209  memory: 2602  loss: 0.8593  loss_cls: 0.4716  loss_bbox: 0.3877
    06/11 16:31:31 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 5/29]  lr: 1.6417e-03  eta: 0:02:38  time: 0.6138  data_time: 0.4256  memory: 2602  loss: 0.8554  loss_cls: 0.4705  loss_bbox: 0.3848
    06/11 16:31:31 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 6/29]  lr: 1.6457e-03  eta: 0:02:36  time: 0.6146  data_time: 0.4268  memory: 2602  loss: 0.8527  loss_cls: 0.4679  loss_bbox: 0.3848
    06/11 16:31:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 7/29]  lr: 1.6497e-03  eta: 0:02:35  time: 0.6138  data_time: 0.4266  memory: 2602  loss: 0.8463  loss_cls: 0.4640  loss_bbox: 0.3823
    06/11 16:31:32 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 8/29]  lr: 1.6537e-03  eta: 0:02:34  time: 0.6162  data_time: 0.4273  memory: 2602  loss: 0.8416  loss_cls: 0.4614  loss_bbox: 0.3802
    06/11 16:31:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][ 9/29]  lr: 1.6577e-03  eta: 0:02:34  time: 0.6197  data_time: 0.4320  memory: 2602  loss: 0.8379  loss_cls: 0.4581  loss_bbox: 0.3798
    06/11 16:31:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][10/29]  lr: 1.6617e-03  eta: 0:02:32  time: 0.6188  data_time: 0.4320  memory: 2606  loss: 0.8353  loss_cls: 0.4560  loss_bbox: 0.3793
    06/11 16:31:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][11/29]  lr: 1.6657e-03  eta: 0:02:31  time: 0.6181  data_time: 0.4314  memory: 2602  loss: 0.8280  loss_cls: 0.4512  loss_bbox: 0.3768
    06/11 16:31:34 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][12/29]  lr: 1.6697e-03  eta: 0:02:30  time: 0.6173  data_time: 0.4309  memory: 2602  loss: 0.8214  loss_cls: 0.4468  loss_bbox: 0.3747
    06/11 16:31:36 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][13/29]  lr: 1.6737e-03  eta: 0:02:29  time: 0.6214  data_time: 0.4377  memory: 2602  loss: 0.8208  loss_cls: 0.4473  loss_bbox: 0.3735
    06/11 16:31:36 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][14/29]  lr: 1.6777e-03  eta: 0:02:28  time: 0.6176  data_time: 0.4359  memory: 2602  loss: 0.8196  loss_cls: 0.4449  loss_bbox: 0.3748
    06/11 16:31:37 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][15/29]  lr: 1.6817e-03  eta: 0:02:27  time: 0.6177  data_time: 0.4372  memory: 2602  loss: 0.8183  loss_cls: 0.4445  loss_bbox: 0.3738
    06/11 16:31:37 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][16/29]  lr: 1.6857e-03  eta: 0:02:26  time: 0.6175  data_time: 0.4372  memory: 2602  loss: 0.8158  loss_cls: 0.4427  loss_bbox: 0.3732
    06/11 16:31:39 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][17/29]  lr: 1.6897e-03  eta: 0:02:25  time: 0.6327  data_time: 0.4530  memory: 2602  loss: 0.8134  loss_cls: 0.4407  loss_bbox: 0.3726
    06/11 16:31:39 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][18/29]  lr: 1.6937e-03  eta: 0:02:24  time: 0.6331  data_time: 0.4536  memory: 2602  loss: 0.8138  loss_cls: 0.4404  loss_bbox: 0.3733
    06/11 16:31:39 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][19/29]  lr: 1.6977e-03  eta: 0:02:23  time: 0.6355  data_time: 0.4538  memory: 2602  loss: 0.8100  loss_cls: 0.4378  loss_bbox: 0.3723
    06/11 16:31:40 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][20/29]  lr: 1.7017e-03  eta: 0:02:22  time: 0.6361  data_time: 0.4540  memory: 2602  loss: 0.8090  loss_cls: 0.4376  loss_bbox: 0.3714
    06/11 16:31:41 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][21/29]  lr: 1.7057e-03  eta: 0:02:21  time: 0.6598  data_time: 0.4758  memory: 2602  loss: 0.8065  loss_cls: 0.4344  loss_bbox: 0.3721
    06/11 16:31:42 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][22/29]  lr: 1.7097e-03  eta: 0:02:20  time: 0.6051  data_time: 0.4228  memory: 2602  loss: 0.8022  loss_cls: 0.4311  loss_bbox: 0.3711
    06/11 16:31:42 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][23/29]  lr: 1.7137e-03  eta: 0:02:19  time: 0.6043  data_time: 0.4222  memory: 2602  loss: 0.8001  loss_cls: 0.4281  loss_bbox: 0.3720
    06/11 16:31:42 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][24/29]  lr: 1.7177e-03  eta: 0:02:17  time: 0.6057  data_time: 0.4231  memory: 2602  loss: 0.7956  loss_cls: 0.4255  loss_bbox: 0.3701
    06/11 16:31:44 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][25/29]  lr: 1.7217e-03  eta: 0:02:17  time: 0.6322  data_time: 0.4498  memory: 2602  loss: 0.7969  loss_cls: 0.4268  loss_bbox: 0.3700
    06/11 16:31:44 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][26/29]  lr: 1.7257e-03  eta: 0:02:16  time: 0.6078  data_time: 0.4256  memory: 2602  loss: 0.7957  loss_cls: 0.4253  loss_bbox: 0.3704
    06/11 16:31:44 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][27/29]  lr: 1.7298e-03  eta: 0:02:15  time: 0.6091  data_time: 0.4252  memory: 2602  loss: 0.7953  loss_cls: 0.4244  loss_bbox: 0.3709
    06/11 16:31:44 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][28/29]  lr: 1.7338e-03  eta: 0:02:13  time: 0.6101  data_time: 0.4251  memory: 2602  loss: 0.7928  loss_cls: 0.4232  loss_bbox: 0.3696
    06/11 16:31:45 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:31:45 - mmengine - [4m[97mINFO[0m - Epoch(train) [15][29/29]  lr: 1.7378e-03  eta: 0:02:12  time: 0.6103  data_time: 0.4256  memory: 1348  loss: 0.7904  loss_cls: 0.4221  loss_bbox: 0.3683
    06/11 16:31:46 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 1/14]    eta: 0:00:19  time: 0.6642  data_time: 0.5467  memory: 245  
    06/11 16:31:46 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 2/14]    eta: 0:00:09  time: 0.6631  data_time: 0.5462  memory: 245  
    06/11 16:31:48 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 3/14]    eta: 0:00:10  time: 0.6639  data_time: 0.5466  memory: 245  
    06/11 16:31:48 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 4/14]    eta: 0:00:07  time: 0.6641  data_time: 0.5466  memory: 245  
    06/11 16:31:49 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 5/14]    eta: 0:00:07  time: 0.6654  data_time: 0.5476  memory: 245  
    06/11 16:31:49 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 6/14]    eta: 0:00:05  time: 0.6620  data_time: 0.5450  memory: 245  
    06/11 16:31:50 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 7/14]    eta: 0:00:05  time: 0.6689  data_time: 0.5524  memory: 245  
    06/11 16:31:50 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 8/14]    eta: 0:00:04  time: 0.6610  data_time: 0.5454  memory: 245  
    06/11 16:31:51 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][ 9/14]    eta: 0:00:03  time: 0.6569  data_time: 0.5405  memory: 245  
    06/11 16:31:52 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][10/14]    eta: 0:00:02  time: 0.6573  data_time: 0.5405  memory: 245  
    06/11 16:31:53 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][11/14]    eta: 0:00:02  time: 0.6523  data_time: 0.5366  memory: 245  
    06/11 16:31:53 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][12/14]    eta: 0:00:01  time: 0.6517  data_time: 0.5365  memory: 245  
    06/11 16:31:54 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][13/14]    eta: 0:00:00  time: 0.6496  data_time: 0.5354  memory: 245  
    06/11 16:31:54 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][14/14]    eta: 0:00:00  time: 0.6524  data_time: 0.5386  memory: 245  
    06/11 16:31:54 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.52s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.530
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.867
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.588
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.530
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.575
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.697
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.699
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699
    06/11 16:31:55 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.530 0.867 0.588 -1.000 -1.000 0.530
    06/11 16:31:55 - mmengine - [4m[97mINFO[0m - Epoch(val) [15][14/14]    coco/bbox_mAP: 0.5300  coco/bbox_mAP_50: 0.8670  coco/bbox_mAP_75: 0.5880  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.5300  data_time: 0.5432  time: 0.6465
    06/11 16:31:55 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_14.pth is removed
    06/11 16:31:59 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.5300 coco/bbox_mAP at 15 epoch is saved to best_coco_bbox_mAP_epoch_15.pth.
    06/11 16:32:06 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 1/29]  lr: 1.7418e-03  eta: 0:02:12  time: 0.6343  data_time: 0.4470  memory: 2602  loss: 0.7852  loss_cls: 0.4177  loss_bbox: 0.3674
    06/11 16:32:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 2/29]  lr: 1.7458e-03  eta: 0:02:11  time: 0.6350  data_time: 0.4471  memory: 2602  loss: 0.7862  loss_cls: 0.4178  loss_bbox: 0.3684
    06/11 16:32:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 3/29]  lr: 1.7498e-03  eta: 0:02:10  time: 0.6372  data_time: 0.4487  memory: 2602  loss: 0.7811  loss_cls: 0.4158  loss_bbox: 0.3653
    06/11 16:32:07 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 4/29]  lr: 1.7538e-03  eta: 0:02:09  time: 0.6394  data_time: 0.4500  memory: 2602  loss: 0.7806  loss_cls: 0.4161  loss_bbox: 0.3646
    06/11 16:32:09 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 5/29]  lr: 1.7578e-03  eta: 0:02:08  time: 0.6338  data_time: 0.4451  memory: 2602  loss: 0.7816  loss_cls: 0.4164  loss_bbox: 0.3652
    06/11 16:32:09 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 6/29]  lr: 1.7618e-03  eta: 0:02:07  time: 0.6342  data_time: 0.4460  memory: 2602  loss: 0.7802  loss_cls: 0.4161  loss_bbox: 0.3641
    06/11 16:32:09 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 7/29]  lr: 1.7658e-03  eta: 0:02:06  time: 0.6337  data_time: 0.4456  memory: 2602  loss: 0.7776  loss_cls: 0.4148  loss_bbox: 0.3628
    06/11 16:32:09 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 8/29]  lr: 1.7698e-03  eta: 0:02:04  time: 0.6324  data_time: 0.4445  memory: 2602  loss: 0.7740  loss_cls: 0.4133  loss_bbox: 0.3608
    06/11 16:32:11 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][ 9/29]  lr: 1.7738e-03  eta: 0:02:04  time: 0.6375  data_time: 0.4484  memory: 2603  loss: 0.7724  loss_cls: 0.4114  loss_bbox: 0.3610
    06/11 16:32:12 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][10/29]  lr: 1.7778e-03  eta: 0:02:03  time: 0.6377  data_time: 0.4484  memory: 2602  loss: 0.7708  loss_cls: 0.4104  loss_bbox: 0.3605
    06/11 16:32:12 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][11/29]  lr: 1.7818e-03  eta: 0:02:02  time: 0.6413  data_time: 0.4508  memory: 2602  loss: 0.7701  loss_cls: 0.4089  loss_bbox: 0.3612
    06/11 16:32:12 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][12/29]  lr: 1.7858e-03  eta: 0:02:01  time: 0.6416  data_time: 0.4514  memory: 2602  loss: 0.7711  loss_cls: 0.4094  loss_bbox: 0.3617
    06/11 16:32:14 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][13/29]  lr: 1.7898e-03  eta: 0:02:00  time: 0.6494  data_time: 0.4579  memory: 2602  loss: 0.7710  loss_cls: 0.4089  loss_bbox: 0.3621
    06/11 16:32:14 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][14/29]  lr: 1.7938e-03  eta: 0:01:59  time: 0.6503  data_time: 0.4586  memory: 2602  loss: 0.7663  loss_cls: 0.4059  loss_bbox: 0.3605
    06/11 16:32:14 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][15/29]  lr: 1.7978e-03  eta: 0:01:58  time: 0.6509  data_time: 0.4590  memory: 2602  loss: 0.7620  loss_cls: 0.4025  loss_bbox: 0.3595
    06/11 16:32:15 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][16/29]  lr: 1.8018e-03  eta: 0:01:57  time: 0.6514  data_time: 0.4591  memory: 2602  loss: 0.7595  loss_cls: 0.4001  loss_bbox: 0.3594
    06/11 16:32:16 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][17/29]  lr: 1.8058e-03  eta: 0:01:56  time: 0.6488  data_time: 0.4565  memory: 2602  loss: 0.7558  loss_cls: 0.3987  loss_bbox: 0.3571
    06/11 16:32:17 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][18/29]  lr: 1.8098e-03  eta: 0:01:55  time: 0.6494  data_time: 0.4568  memory: 2602  loss: 0.7521  loss_cls: 0.3956  loss_bbox: 0.3565
    06/11 16:32:17 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][19/29]  lr: 1.8138e-03  eta: 0:01:54  time: 0.6497  data_time: 0.4570  memory: 2602  loss: 0.7505  loss_cls: 0.3943  loss_bbox: 0.3562
    06/11 16:32:17 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][20/29]  lr: 1.8178e-03  eta: 0:01:53  time: 0.6514  data_time: 0.4582  memory: 2602  loss: 0.7480  loss_cls: 0.3930  loss_bbox: 0.3550
    06/11 16:32:19 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][21/29]  lr: 1.8218e-03  eta: 0:01:52  time: 0.6774  data_time: 0.4837  memory: 2602  loss: 0.7455  loss_cls: 0.3902  loss_bbox: 0.3552
    06/11 16:32:19 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][22/29]  lr: 1.8258e-03  eta: 0:01:51  time: 0.6315  data_time: 0.4385  memory: 2602  loss: 0.7428  loss_cls: 0.3896  loss_bbox: 0.3531
    06/11 16:32:19 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][23/29]  lr: 1.8299e-03  eta: 0:01:50  time: 0.6310  data_time: 0.4382  memory: 2602  loss: 0.7414  loss_cls: 0.3876  loss_bbox: 0.3537
    06/11 16:32:19 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][24/29]  lr: 1.8339e-03  eta: 0:01:49  time: 0.6314  data_time: 0.4384  memory: 2602  loss: 0.7368  loss_cls: 0.3822  loss_bbox: 0.3546
    06/11 16:32:21 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][25/29]  lr: 1.8379e-03  eta: 0:01:48  time: 0.6602  data_time: 0.4662  memory: 2602  loss: 0.7345  loss_cls: 0.3792  loss_bbox: 0.3553
    06/11 16:32:21 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][26/29]  lr: 1.8419e-03  eta: 0:01:47  time: 0.6255  data_time: 0.4325  memory: 2602  loss: 0.7320  loss_cls: 0.3769  loss_bbox: 0.3550
    06/11 16:32:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][27/29]  lr: 1.8459e-03  eta: 0:01:46  time: 0.6279  data_time: 0.4340  memory: 2602  loss: 0.7303  loss_cls: 0.3752  loss_bbox: 0.3551
    06/11 16:32:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][28/29]  lr: 1.8499e-03  eta: 0:01:45  time: 0.6270  data_time: 0.4336  memory: 2602  loss: 0.7291  loss_cls: 0.3735  loss_bbox: 0.3555
    06/11 16:32:22 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:32:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [16][29/29]  lr: 1.8539e-03  eta: 0:01:44  time: 0.6241  data_time: 0.4334  memory: 1345  loss: 0.7310  loss_cls: 0.3749  loss_bbox: 0.3560
    06/11 16:32:23 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 1/14]    eta: 0:00:16  time: 0.6536  data_time: 0.5402  memory: 245  
    06/11 16:32:23 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 2/14]    eta: 0:00:08  time: 0.6543  data_time: 0.5412  memory: 245  
    06/11 16:32:24 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 3/14]    eta: 0:00:08  time: 0.6505  data_time: 0.5379  memory: 245  
    06/11 16:32:25 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 4/14]    eta: 0:00:06  time: 0.6552  data_time: 0.5433  memory: 245  
    06/11 16:32:26 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 5/14]    eta: 0:00:06  time: 0.6528  data_time: 0.5421  memory: 245  
    06/11 16:32:26 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 6/14]    eta: 0:00:05  time: 0.6551  data_time: 0.5453  memory: 245  
    06/11 16:32:27 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 7/14]    eta: 0:00:05  time: 0.6574  data_time: 0.5491  memory: 245  
    06/11 16:32:27 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 8/14]    eta: 0:00:04  time: 0.6622  data_time: 0.5533  memory: 245  
    06/11 16:32:28 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][ 9/14]    eta: 0:00:03  time: 0.6548  data_time: 0.5464  memory: 245  
    06/11 16:32:29 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][10/14]    eta: 0:00:02  time: 0.6530  data_time: 0.5448  memory: 245  
    06/11 16:32:30 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][11/14]    eta: 0:00:02  time: 0.6528  data_time: 0.5451  memory: 245  
    06/11 16:32:30 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][12/14]    eta: 0:00:01  time: 0.6568  data_time: 0.5487  memory: 245  
    06/11 16:32:31 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][13/14]    eta: 0:00:00  time: 0.6499  data_time: 0.5422  memory: 245  
    06/11 16:32:31 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][14/14]    eta: 0:00:00  time: 0.6601  data_time: 0.5529  memory: 245  
    06/11 16:32:32 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.26s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.51s).
    Accumulating evaluation results...
    DONE (t=0.23s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.949
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.779
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.629
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.650
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.715
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.715
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.715
    06/11 16:32:33 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.629 0.949 0.779 -1.000 -1.000 0.629
    06/11 16:32:33 - mmengine - [4m[97mINFO[0m - Epoch(val) [16][14/14]    coco/bbox_mAP: 0.6290  coco/bbox_mAP_50: 0.9490  coco/bbox_mAP_75: 0.7790  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.6290  data_time: 0.5544  time: 0.6438
    06/11 16:32:33 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_15.pth is removed
    06/11 16:32:36 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.6290 coco/bbox_mAP at 16 epoch is saved to best_coco_bbox_mAP_epoch_16.pth.
    06/11 16:32:43 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 1/29]  lr: 1.8579e-03  eta: 0:01:43  time: 0.6419  data_time: 0.4515  memory: 2602  loss: 0.7275  loss_cls: 0.3722  loss_bbox: 0.3552
    06/11 16:32:43 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 2/29]  lr: 1.8619e-03  eta: 0:01:42  time: 0.6424  data_time: 0.4517  memory: 2602  loss: 0.7254  loss_cls: 0.3702  loss_bbox: 0.3551
    06/11 16:32:43 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 3/29]  lr: 1.8659e-03  eta: 0:01:41  time: 0.6429  data_time: 0.4517  memory: 2602  loss: 0.7258  loss_cls: 0.3696  loss_bbox: 0.3562
    06/11 16:32:43 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 4/29]  lr: 1.8699e-03  eta: 0:01:40  time: 0.6447  data_time: 0.4530  memory: 2602  loss: 0.7272  loss_cls: 0.3694  loss_bbox: 0.3578
    06/11 16:32:45 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 5/29]  lr: 1.8739e-03  eta: 0:01:39  time: 0.6462  data_time: 0.4518  memory: 2602  loss: 0.7211  loss_cls: 0.3647  loss_bbox: 0.3563
    06/11 16:32:45 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 6/29]  lr: 1.8779e-03  eta: 0:01:38  time: 0.6481  data_time: 0.4520  memory: 2602  loss: 0.7155  loss_cls: 0.3625  loss_bbox: 0.3531
    06/11 16:32:46 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 7/29]  lr: 1.8819e-03  eta: 0:01:37  time: 0.6466  data_time: 0.4507  memory: 2602  loss: 0.7101  loss_cls: 0.3590  loss_bbox: 0.3511
    06/11 16:32:46 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 8/29]  lr: 1.8859e-03  eta: 0:01:36  time: 0.6478  data_time: 0.4518  memory: 2602  loss: 0.7056  loss_cls: 0.3561  loss_bbox: 0.3495
    06/11 16:32:47 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][ 9/29]  lr: 1.8899e-03  eta: 0:01:35  time: 0.6355  data_time: 0.4400  memory: 2602  loss: 0.7020  loss_cls: 0.3533  loss_bbox: 0.3487
    06/11 16:32:48 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][10/29]  lr: 1.8939e-03  eta: 0:01:34  time: 0.6346  data_time: 0.4393  memory: 2602  loss: 0.6937  loss_cls: 0.3482  loss_bbox: 0.3456
    06/11 16:32:48 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][11/29]  lr: 1.8979e-03  eta: 0:01:33  time: 0.6421  data_time: 0.4470  memory: 2602  loss: 0.6911  loss_cls: 0.3463  loss_bbox: 0.3449
    06/11 16:32:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][12/29]  lr: 1.9019e-03  eta: 0:01:32  time: 0.6430  data_time: 0.4475  memory: 2602  loss: 0.6859  loss_cls: 0.3428  loss_bbox: 0.3432
    06/11 16:32:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][13/29]  lr: 1.9059e-03  eta: 0:01:32  time: 0.6399  data_time: 0.4458  memory: 2602  loss: 0.6807  loss_cls: 0.3402  loss_bbox: 0.3405
    06/11 16:32:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][14/29]  lr: 1.9099e-03  eta: 0:01:31  time: 0.6407  data_time: 0.4470  memory: 2602  loss: 0.6793  loss_cls: 0.3396  loss_bbox: 0.3397
    06/11 16:32:51 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][15/29]  lr: 1.9139e-03  eta: 0:01:30  time: 0.6440  data_time: 0.4501  memory: 2602  loss: 0.6768  loss_cls: 0.3386  loss_bbox: 0.3382
    06/11 16:32:51 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][16/29]  lr: 1.9179e-03  eta: 0:01:29  time: 0.6416  data_time: 0.4486  memory: 2602  loss: 0.6774  loss_cls: 0.3397  loss_bbox: 0.3377
    06/11 16:32:52 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][17/29]  lr: 1.9219e-03  eta: 0:01:28  time: 0.6338  data_time: 0.4400  memory: 2602  loss: 0.6726  loss_cls: 0.3356  loss_bbox: 0.3370
    06/11 16:32:52 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][18/29]  lr: 1.9259e-03  eta: 0:01:27  time: 0.6328  data_time: 0.4399  memory: 2602  loss: 0.6687  loss_cls: 0.3333  loss_bbox: 0.3354
    06/11 16:32:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][19/29]  lr: 1.9300e-03  eta: 0:01:26  time: 0.6391  data_time: 0.4464  memory: 2602  loss: 0.6703  loss_cls: 0.3336  loss_bbox: 0.3367
    06/11 16:32:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][20/29]  lr: 1.9340e-03  eta: 0:01:25  time: 0.6397  data_time: 0.4465  memory: 2602  loss: 0.6669  loss_cls: 0.3320  loss_bbox: 0.3349
    06/11 16:32:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][21/29]  lr: 1.9380e-03  eta: 0:01:24  time: 0.6618  data_time: 0.4662  memory: 2602  loss: 0.6604  loss_cls: 0.3270  loss_bbox: 0.3334
    06/11 16:32:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][22/29]  lr: 1.9420e-03  eta: 0:01:23  time: 0.6091  data_time: 0.4149  memory: 2602  loss: 0.6592  loss_cls: 0.3252  loss_bbox: 0.3339
    06/11 16:32:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][23/29]  lr: 1.9460e-03  eta: 0:01:22  time: 0.6099  data_time: 0.4161  memory: 2602  loss: 0.6549  loss_cls: 0.3228  loss_bbox: 0.3321
    06/11 16:32:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][24/29]  lr: 1.9500e-03  eta: 0:01:21  time: 0.6081  data_time: 0.4140  memory: 2602  loss: 0.6539  loss_cls: 0.3217  loss_bbox: 0.3322
    06/11 16:32:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][25/29]  lr: 1.9540e-03  eta: 0:01:20  time: 0.6293  data_time: 0.4346  memory: 2602  loss: 0.6488  loss_cls: 0.3174  loss_bbox: 0.3314
    06/11 16:32:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][26/29]  lr: 1.9580e-03  eta: 0:01:19  time: 0.6055  data_time: 0.4120  memory: 2602  loss: 0.6435  loss_cls: 0.3129  loss_bbox: 0.3306
    06/11 16:32:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][27/29]  lr: 1.9620e-03  eta: 0:01:18  time: 0.6050  data_time: 0.4115  memory: 2602  loss: 0.6427  loss_cls: 0.3126  loss_bbox: 0.3301
    06/11 16:32:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][28/29]  lr: 1.9660e-03  eta: 0:01:17  time: 0.6056  data_time: 0.4126  memory: 2602  loss: 0.6405  loss_cls: 0.3106  loss_bbox: 0.3299
    06/11 16:32:58 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:32:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [17][29/29]  lr: 1.9700e-03  eta: 0:01:16  time: 0.6099  data_time: 0.4173  memory: 1345  loss: 0.6413  loss_cls: 0.3107  loss_bbox: 0.3306
    06/11 16:32:59 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 1/14]    eta: 0:00:19  time: 0.6637  data_time: 0.5566  memory: 245  
    06/11 16:33:00 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 2/14]    eta: 0:00:09  time: 0.6638  data_time: 0.5567  memory: 245  
    06/11 16:33:01 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 3/14]    eta: 0:00:10  time: 0.6628  data_time: 0.5557  memory: 245  
    06/11 16:33:01 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 4/14]    eta: 0:00:07  time: 0.6626  data_time: 0.5558  memory: 245  
    06/11 16:33:02 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 5/14]    eta: 0:00:07  time: 0.6671  data_time: 0.5599  memory: 245  
    06/11 16:33:02 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 6/14]    eta: 0:00:05  time: 0.6634  data_time: 0.5572  memory: 245  
    06/11 16:33:03 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 7/14]    eta: 0:00:05  time: 0.6695  data_time: 0.5640  memory: 245  
    06/11 16:33:03 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 8/14]    eta: 0:00:04  time: 0.6649  data_time: 0.5598  memory: 245  
    06/11 16:33:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][ 9/14]    eta: 0:00:03  time: 0.6605  data_time: 0.5558  memory: 245  
    06/11 16:33:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][10/14]    eta: 0:00:02  time: 0.6589  data_time: 0.5556  memory: 245  
    06/11 16:33:06 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][11/14]    eta: 0:00:02  time: 0.6574  data_time: 0.5551  memory: 245  
    06/11 16:33:06 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][12/14]    eta: 0:00:01  time: 0.6565  data_time: 0.5550  memory: 245  
    06/11 16:33:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][13/14]    eta: 0:00:00  time: 0.6541  data_time: 0.5537  memory: 245  
    06/11 16:33:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][14/14]    eta: 0:00:00  time: 0.6573  data_time: 0.5573  memory: 245  
    06/11 16:33:07 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.15s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.50s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.610
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.944
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.725
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.610
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.636
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.714
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.715
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.715
    06/11 16:33:08 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.610 0.944 0.725 -1.000 -1.000 0.610
    06/11 16:33:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [17][14/14]    coco/bbox_mAP: 0.6100  coco/bbox_mAP_50: 0.9440  coco/bbox_mAP_75: 0.7250  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.6100  data_time: 0.5616  time: 0.6615
    06/11 16:33:11 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 1/29]  lr: 1.9740e-03  eta: 0:01:16  time: 0.6160  data_time: 0.4251  memory: 2602  loss: 0.6368  loss_cls: 0.3083  loss_bbox: 0.3285
    06/11 16:33:11 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 2/29]  lr: 1.9780e-03  eta: 0:01:15  time: 0.6160  data_time: 0.4256  memory: 2602  loss: 0.6364  loss_cls: 0.3073  loss_bbox: 0.3291
    06/11 16:33:11 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 3/29]  lr: 1.9820e-03  eta: 0:01:14  time: 0.6130  data_time: 0.4233  memory: 2602  loss: 0.6362  loss_cls: 0.3054  loss_bbox: 0.3308
    06/11 16:33:11 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 4/29]  lr: 1.9860e-03  eta: 0:01:13  time: 0.6138  data_time: 0.4225  memory: 2603  loss: 0.6361  loss_cls: 0.3053  loss_bbox: 0.3308
    06/11 16:33:13 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 5/29]  lr: 1.9900e-03  eta: 0:01:12  time: 0.6142  data_time: 0.4222  memory: 2602  loss: 0.6364  loss_cls: 0.3050  loss_bbox: 0.3314
    06/11 16:33:14 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 6/29]  lr: 1.9940e-03  eta: 0:01:11  time: 0.6139  data_time: 0.4214  memory: 2602  loss: 0.6347  loss_cls: 0.3031  loss_bbox: 0.3317
    06/11 16:33:14 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 7/29]  lr: 1.9980e-03  eta: 0:01:10  time: 0.6136  data_time: 0.4211  memory: 2602  loss: 0.6330  loss_cls: 0.3024  loss_bbox: 0.3307
    06/11 16:33:14 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 8/29]  lr: 2.0020e-03  eta: 0:01:09  time: 0.6159  data_time: 0.4236  memory: 2602  loss: 0.6318  loss_cls: 0.3020  loss_bbox: 0.3298
    06/11 16:33:16 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][ 9/29]  lr: 2.0060e-03  eta: 0:01:08  time: 0.6135  data_time: 0.4208  memory: 2602  loss: 0.6291  loss_cls: 0.3000  loss_bbox: 0.3291
    06/11 16:33:16 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][10/29]  lr: 2.0100e-03  eta: 0:01:07  time: 0.6153  data_time: 0.4230  memory: 2602  loss: 0.6304  loss_cls: 0.3003  loss_bbox: 0.3301
    06/11 16:33:16 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][11/29]  lr: 2.0140e-03  eta: 0:01:06  time: 0.6145  data_time: 0.4226  memory: 2602  loss: 0.6287  loss_cls: 0.2985  loss_bbox: 0.3302
    06/11 16:33:16 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][12/29]  lr: 2.0180e-03  eta: 0:01:05  time: 0.6119  data_time: 0.4208  memory: 2602  loss: 0.6286  loss_cls: 0.2968  loss_bbox: 0.3318
    06/11 16:33:18 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][13/29]  lr: 2.0220e-03  eta: 0:01:04  time: 0.6137  data_time: 0.4223  memory: 2602  loss: 0.6269  loss_cls: 0.2951  loss_bbox: 0.3318
    06/11 16:33:18 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][14/29]  lr: 2.0260e-03  eta: 0:01:03  time: 0.6140  data_time: 0.4222  memory: 2602  loss: 0.6263  loss_cls: 0.2940  loss_bbox: 0.3323
    06/11 16:33:18 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][15/29]  lr: 2.0300e-03  eta: 0:01:03  time: 0.6143  data_time: 0.4223  memory: 2602  loss: 0.6310  loss_cls: 0.2972  loss_bbox: 0.3339
    06/11 16:33:18 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][16/29]  lr: 2.0341e-03  eta: 0:01:02  time: 0.6127  data_time: 0.4213  memory: 2602  loss: 0.6307  loss_cls: 0.2979  loss_bbox: 0.3328
    06/11 16:33:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][17/29]  lr: 2.0381e-03  eta: 0:01:01  time: 0.6146  data_time: 0.4223  memory: 2602  loss: 0.6299  loss_cls: 0.2975  loss_bbox: 0.3324
    06/11 16:33:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][18/29]  lr: 2.0421e-03  eta: 0:01:00  time: 0.6150  data_time: 0.4232  memory: 2602  loss: 0.6297  loss_cls: 0.2960  loss_bbox: 0.3337
    06/11 16:33:21 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][19/29]  lr: 2.0461e-03  eta: 0:00:59  time: 0.6108  data_time: 0.4201  memory: 2602  loss: 0.6262  loss_cls: 0.2943  loss_bbox: 0.3319
    06/11 16:33:21 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][20/29]  lr: 2.0501e-03  eta: 0:00:58  time: 0.6111  data_time: 0.4198  memory: 2602  loss: 0.6244  loss_cls: 0.2932  loss_bbox: 0.3311
    06/11 16:33:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][21/29]  lr: 2.0541e-03  eta: 0:00:57  time: 0.6349  data_time: 0.4415  memory: 2602  loss: 0.6219  loss_cls: 0.2910  loss_bbox: 0.3309
    06/11 16:33:23 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][22/29]  lr: 2.0581e-03  eta: 0:00:56  time: 0.5851  data_time: 0.3926  memory: 2602  loss: 0.6243  loss_cls: 0.2931  loss_bbox: 0.3311
    06/11 16:33:23 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][23/29]  lr: 2.0621e-03  eta: 0:00:55  time: 0.5845  data_time: 0.3924  memory: 2602  loss: 0.6225  loss_cls: 0.2924  loss_bbox: 0.3301
    06/11 16:33:23 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][24/29]  lr: 2.0661e-03  eta: 0:00:54  time: 0.5877  data_time: 0.3944  memory: 2602  loss: 0.6261  loss_cls: 0.2936  loss_bbox: 0.3324
    06/11 16:33:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][25/29]  lr: 2.0701e-03  eta: 0:00:53  time: 0.6186  data_time: 0.4244  memory: 2602  loss: 0.6255  loss_cls: 0.2933  loss_bbox: 0.3323
    06/11 16:33:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][26/29]  lr: 2.0741e-03  eta: 0:00:53  time: 0.5881  data_time: 0.3959  memory: 2602  loss: 0.6261  loss_cls: 0.2927  loss_bbox: 0.3334
    06/11 16:33:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][27/29]  lr: 2.0781e-03  eta: 0:00:52  time: 0.5860  data_time: 0.3947  memory: 2602  loss: 0.6277  loss_cls: 0.2940  loss_bbox: 0.3337
    06/11 16:33:26 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][28/29]  lr: 2.0821e-03  eta: 0:00:51  time: 0.5872  data_time: 0.3953  memory: 2602  loss: 0.6311  loss_cls: 0.2957  loss_bbox: 0.3354
    06/11 16:33:26 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:33:26 - mmengine - [4m[97mINFO[0m - Epoch(train) [18][29/29]  lr: 2.0861e-03  eta: 0:00:50  time: 0.5903  data_time: 0.3979  memory: 1347  loss: 0.6349  loss_cls: 0.2984  loss_bbox: 0.3365
    06/11 16:33:28 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 1/14]    eta: 0:00:21  time: 0.6695  data_time: 0.5697  memory: 245  
    06/11 16:33:28 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 2/14]    eta: 0:00:10  time: 0.6670  data_time: 0.5680  memory: 245  
    06/11 16:33:29 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 3/14]    eta: 0:00:10  time: 0.6701  data_time: 0.5714  memory: 245  
    06/11 16:33:29 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 4/14]    eta: 0:00:07  time: 0.6701  data_time: 0.5714  memory: 245  
    06/11 16:33:30 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 5/14]    eta: 0:00:07  time: 0.6724  data_time: 0.5743  memory: 245  
    06/11 16:33:30 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 6/14]    eta: 0:00:05  time: 0.6701  data_time: 0.5729  memory: 245  
    06/11 16:33:32 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 7/14]    eta: 0:00:05  time: 0.6782  data_time: 0.5810  memory: 245  
    06/11 16:33:32 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 8/14]    eta: 0:00:04  time: 0.6718  data_time: 0.5747  memory: 245  
    06/11 16:33:33 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][ 9/14]    eta: 0:00:03  time: 0.6654  data_time: 0.5685  memory: 245  
    06/11 16:33:33 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][10/14]    eta: 0:00:02  time: 0.6654  data_time: 0.5687  memory: 245  
    06/11 16:33:34 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][11/14]    eta: 0:00:02  time: 0.6626  data_time: 0.5658  memory: 245  
    06/11 16:33:34 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][12/14]    eta: 0:00:01  time: 0.6629  data_time: 0.5658  memory: 245  
    06/11 16:33:35 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][13/14]    eta: 0:00:00  time: 0.6552  data_time: 0.5586  memory: 245  
    06/11 16:33:35 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][14/14]    eta: 0:00:00  time: 0.6605  data_time: 0.5629  memory: 245  
    06/11 16:33:36 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.05s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.56s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.684
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.963
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.859
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.699
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.743
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.747
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.747
    06/11 16:33:37 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.684 0.963 0.859 -1.000 -1.000 0.684
    06/11 16:33:37 - mmengine - [4m[97mINFO[0m - Epoch(val) [18][14/14]    coco/bbox_mAP: 0.6840  coco/bbox_mAP_50: 0.9630  coco/bbox_mAP_75: 0.8590  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.6840  data_time: 0.5315  time: 0.6352
    06/11 16:33:37 - mmengine - [4m[97mINFO[0m - The previous best checkpoint /public3/labmember/zhengdh/openmmlab-true-files/mmdetection/work_dirs/rtmdet_tiny_drink/best_coco_bbox_mAP_epoch_16.pth is removed
    06/11 16:33:40 - mmengine - [4m[97mINFO[0m - The best checkpoint with 0.6840 coco/bbox_mAP at 18 epoch is saved to best_coco_bbox_mAP_epoch_18.pth.
    06/11 16:33:47 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 1/29]  lr: 2.0901e-03  eta: 0:00:49  time: 0.6130  data_time: 0.4194  memory: 2605  loss: 0.6416  loss_cls: 0.3026  loss_bbox: 0.3390
    06/11 16:33:47 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 2/29]  lr: 2.0941e-03  eta: 0:00:48  time: 0.6149  data_time: 0.4208  memory: 2602  loss: 0.6437  loss_cls: 0.3036  loss_bbox: 0.3401
    06/11 16:33:47 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 3/29]  lr: 2.0981e-03  eta: 0:00:47  time: 0.6061  data_time: 0.4135  memory: 2602  loss: 0.6447  loss_cls: 0.3033  loss_bbox: 0.3414
    06/11 16:33:48 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 4/29]  lr: 2.1021e-03  eta: 0:00:46  time: 0.6049  data_time: 0.4133  memory: 2602  loss: 0.6461  loss_cls: 0.3048  loss_bbox: 0.3413
    06/11 16:33:49 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 5/29]  lr: 2.1061e-03  eta: 0:00:45  time: 0.6036  data_time: 0.4125  memory: 2602  loss: 0.6496  loss_cls: 0.3069  loss_bbox: 0.3426
    06/11 16:33:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 6/29]  lr: 2.1101e-03  eta: 0:00:45  time: 0.6093  data_time: 0.4179  memory: 2602  loss: 0.6478  loss_cls: 0.3054  loss_bbox: 0.3424
    06/11 16:33:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 7/29]  lr: 2.1141e-03  eta: 0:00:44  time: 0.6061  data_time: 0.4152  memory: 2602  loss: 0.6482  loss_cls: 0.3045  loss_bbox: 0.3437
    06/11 16:33:50 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 8/29]  lr: 2.1181e-03  eta: 0:00:43  time: 0.6068  data_time: 0.4157  memory: 2602  loss: 0.6463  loss_cls: 0.3025  loss_bbox: 0.3438
    06/11 16:33:51 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][ 9/29]  lr: 2.1221e-03  eta: 0:00:42  time: 0.6057  data_time: 0.4137  memory: 2602  loss: 0.6466  loss_cls: 0.3025  loss_bbox: 0.3440
    06/11 16:33:52 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][10/29]  lr: 2.1261e-03  eta: 0:00:41  time: 0.6228  data_time: 0.4286  memory: 2602  loss: 0.6474  loss_cls: 0.3032  loss_bbox: 0.3442
    06/11 16:33:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][11/29]  lr: 2.1301e-03  eta: 0:00:40  time: 0.6152  data_time: 0.4221  memory: 2602  loss: 0.6427  loss_cls: 0.3011  loss_bbox: 0.3417
    06/11 16:33:53 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][12/29]  lr: 2.1342e-03  eta: 0:00:39  time: 0.6129  data_time: 0.4215  memory: 2602  loss: 0.6457  loss_cls: 0.3022  loss_bbox: 0.3435
    06/11 16:33:54 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][13/29]  lr: 2.1382e-03  eta: 0:00:38  time: 0.6014  data_time: 0.4112  memory: 2602  loss: 0.6469  loss_cls: 0.3034  loss_bbox: 0.3435
    06/11 16:33:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][14/29]  lr: 2.1422e-03  eta: 0:00:37  time: 0.6196  data_time: 0.4297  memory: 2602  loss: 0.6462  loss_cls: 0.3036  loss_bbox: 0.3426
    06/11 16:33:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][15/29]  lr: 2.1462e-03  eta: 0:00:37  time: 0.6180  data_time: 0.4282  memory: 2602  loss: 0.6483  loss_cls: 0.3047  loss_bbox: 0.3436
    06/11 16:33:55 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][16/29]  lr: 2.1502e-03  eta: 0:00:36  time: 0.6175  data_time: 0.4287  memory: 2602  loss: 0.6463  loss_cls: 0.3036  loss_bbox: 0.3427
    06/11 16:33:56 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][17/29]  lr: 2.1542e-03  eta: 0:00:35  time: 0.6007  data_time: 0.4127  memory: 2602  loss: 0.6486  loss_cls: 0.3055  loss_bbox: 0.3431
    06/11 16:33:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][18/29]  lr: 2.1582e-03  eta: 0:00:34  time: 0.6270  data_time: 0.4390  memory: 2602  loss: 0.6493  loss_cls: 0.3073  loss_bbox: 0.3420
    06/11 16:33:57 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][19/29]  lr: 2.1622e-03  eta: 0:00:33  time: 0.6264  data_time: 0.4385  memory: 2602  loss: 0.6480  loss_cls: 0.3060  loss_bbox: 0.3420
    06/11 16:33:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][20/29]  lr: 2.1662e-03  eta: 0:00:32  time: 0.6257  data_time: 0.4376  memory: 2602  loss: 0.6478  loss_cls: 0.3057  loss_bbox: 0.3421
    06/11 16:33:58 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][21/29]  lr: 2.1702e-03  eta: 0:00:31  time: 0.6265  data_time: 0.4365  memory: 2602  loss: 0.6467  loss_cls: 0.3042  loss_bbox: 0.3425
    06/11 16:33:59 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][22/29]  lr: 2.1742e-03  eta: 0:00:30  time: 0.5973  data_time: 0.4069  memory: 2602  loss: 0.6462  loss_cls: 0.3037  loss_bbox: 0.3425
    06/11 16:33:59 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][23/29]  lr: 2.1782e-03  eta: 0:00:29  time: 0.5979  data_time: 0.4072  memory: 2602  loss: 0.6449  loss_cls: 0.3029  loss_bbox: 0.3419
    06/11 16:33:59 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][24/29]  lr: 2.1822e-03  eta: 0:00:29  time: 0.5965  data_time: 0.4063  memory: 2603  loss: 0.6437  loss_cls: 0.3024  loss_bbox: 0.3413
    06/11 16:34:00 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][25/29]  lr: 2.1862e-03  eta: 0:00:28  time: 0.6088  data_time: 0.4190  memory: 2602  loss: 0.6418  loss_cls: 0.3007  loss_bbox: 0.3411
    06/11 16:34:01 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][26/29]  lr: 2.1902e-03  eta: 0:00:27  time: 0.5836  data_time: 0.3953  memory: 2602  loss: 0.6378  loss_cls: 0.2989  loss_bbox: 0.3388
    06/11 16:34:01 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][27/29]  lr: 2.1942e-03  eta: 0:00:26  time: 0.5832  data_time: 0.3956  memory: 2602  loss: 0.6390  loss_cls: 0.3003  loss_bbox: 0.3386
    06/11 16:34:01 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][28/29]  lr: 2.1982e-03  eta: 0:00:25  time: 0.5840  data_time: 0.3962  memory: 2602  loss: 0.6390  loss_cls: 0.2999  loss_bbox: 0.3391
    06/11 16:34:02 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:34:02 - mmengine - [4m[97mINFO[0m - Epoch(train) [19][29/29]  lr: 2.2022e-03  eta: 0:00:24  time: 0.5807  data_time: 0.3937  memory: 1345  loss: 0.6392  loss_cls: 0.2995  loss_bbox: 0.3397
    06/11 16:34:03 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 1/14]    eta: 0:00:17  time: 0.6625  data_time: 0.5646  memory: 245  
    06/11 16:34:03 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 2/14]    eta: 0:00:08  time: 0.6634  data_time: 0.5652  memory: 245  
    06/11 16:34:04 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 3/14]    eta: 0:00:09  time: 0.6615  data_time: 0.5634  memory: 245  
    06/11 16:34:04 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 4/14]    eta: 0:00:06  time: 0.6616  data_time: 0.5634  memory: 245  
    06/11 16:34:05 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 5/14]    eta: 0:00:06  time: 0.6641  data_time: 0.5661  memory: 245  
    06/11 16:34:06 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 6/14]    eta: 0:00:05  time: 0.6637  data_time: 0.5661  memory: 245  
    06/11 16:34:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 7/14]    eta: 0:00:05  time: 0.6694  data_time: 0.5714  memory: 245  
    06/11 16:34:07 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 8/14]    eta: 0:00:04  time: 0.6654  data_time: 0.5688  memory: 245  
    06/11 16:34:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][ 9/14]    eta: 0:00:03  time: 0.6632  data_time: 0.5664  memory: 245  
    06/11 16:34:08 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][10/14]    eta: 0:00:02  time: 0.6620  data_time: 0.5655  memory: 245  
    06/11 16:34:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][11/14]    eta: 0:00:02  time: 0.6626  data_time: 0.5665  memory: 245  
    06/11 16:34:09 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][12/14]    eta: 0:00:01  time: 0.6576  data_time: 0.5611  memory: 245  
    06/11 16:34:10 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][13/14]    eta: 0:00:00  time: 0.6609  data_time: 0.5640  memory: 245  
    06/11 16:34:11 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][14/14]    eta: 0:00:00  time: 0.6607  data_time: 0.5629  memory: 245  
    06/11 16:34:11 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.51s).
    Accumulating evaluation results...
    DONE (t=0.16s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.636
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.973
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.822
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.636
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.664
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.699
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.699
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.699
    06/11 16:34:12 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.636 0.973 0.822 -1.000 -1.000 0.636
    06/11 16:34:12 - mmengine - [4m[97mINFO[0m - Epoch(val) [19][14/14]    coco/bbox_mAP: 0.6360  coco/bbox_mAP_50: 0.9730  coco/bbox_mAP_75: 0.8220  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.6360  data_time: 0.5321  time: 0.6298
    06/11 16:34:15 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 1/29]  lr: 2.2062e-03  eta: 0:00:23  time: 0.6064  data_time: 0.4190  memory: 2603  loss: 0.6417  loss_cls: 0.2997  loss_bbox: 0.3420
    06/11 16:34:15 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 2/29]  lr: 2.2102e-03  eta: 0:00:23  time: 0.6097  data_time: 0.4208  memory: 2602  loss: 0.6411  loss_cls: 0.2993  loss_bbox: 0.3418
    06/11 16:34:15 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 3/29]  lr: 2.2142e-03  eta: 0:00:22  time: 0.6114  data_time: 0.4215  memory: 2602  loss: 0.6410  loss_cls: 0.2994  loss_bbox: 0.3416
    06/11 16:34:16 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 4/29]  lr: 2.2182e-03  eta: 0:00:21  time: 0.6128  data_time: 0.4215  memory: 2602  loss: 0.6390  loss_cls: 0.2990  loss_bbox: 0.3399
    06/11 16:34:17 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 5/29]  lr: 2.2222e-03  eta: 0:00:20  time: 0.6077  data_time: 0.4166  memory: 2602  loss: 0.6391  loss_cls: 0.2985  loss_bbox: 0.3406
    06/11 16:34:17 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 6/29]  lr: 2.2262e-03  eta: 0:00:19  time: 0.6101  data_time: 0.4189  memory: 2602  loss: 0.6373  loss_cls: 0.2966  loss_bbox: 0.3407
    06/11 16:34:18 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 7/29]  lr: 2.2302e-03  eta: 0:00:18  time: 0.6098  data_time: 0.4188  memory: 2602  loss: 0.6322  loss_cls: 0.2928  loss_bbox: 0.3394
    06/11 16:34:18 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 8/29]  lr: 2.2343e-03  eta: 0:00:17  time: 0.6103  data_time: 0.4190  memory: 2602  loss: 0.6304  loss_cls: 0.2911  loss_bbox: 0.3393
    06/11 16:34:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][ 9/29]  lr: 2.2383e-03  eta: 0:00:17  time: 0.6127  data_time: 0.4210  memory: 2602  loss: 0.6302  loss_cls: 0.2904  loss_bbox: 0.3398
    06/11 16:34:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][10/29]  lr: 2.2423e-03  eta: 0:00:16  time: 0.6180  data_time: 0.4255  memory: 2602  loss: 0.6308  loss_cls: 0.2903  loss_bbox: 0.3405
    06/11 16:34:20 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][11/29]  lr: 2.2463e-03  eta: 0:00:15  time: 0.6172  data_time: 0.4250  memory: 2602  loss: 0.6321  loss_cls: 0.2900  loss_bbox: 0.3421
    06/11 16:34:21 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][12/29]  lr: 2.2503e-03  eta: 0:00:14  time: 0.6178  data_time: 0.4260  memory: 2602  loss: 0.6329  loss_cls: 0.2891  loss_bbox: 0.3438
    06/11 16:34:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][13/29]  lr: 2.2543e-03  eta: 0:00:13  time: 0.6206  data_time: 0.4285  memory: 2602  loss: 0.6342  loss_cls: 0.2889  loss_bbox: 0.3453
    06/11 16:34:22 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][14/29]  lr: 2.2583e-03  eta: 0:00:12  time: 0.6207  data_time: 0.4284  memory: 2602  loss: 0.6327  loss_cls: 0.2868  loss_bbox: 0.3459
    06/11 16:34:23 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][15/29]  lr: 2.2623e-03  eta: 0:00:11  time: 0.6209  data_time: 0.4285  memory: 2602  loss: 0.6347  loss_cls: 0.2885  loss_bbox: 0.3462
    06/11 16:34:23 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][16/29]  lr: 2.2663e-03  eta: 0:00:11  time: 0.6188  data_time: 0.4268  memory: 2602  loss: 0.6352  loss_cls: 0.2897  loss_bbox: 0.3455
    06/11 16:34:24 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][17/29]  lr: 2.2703e-03  eta: 0:00:10  time: 0.6143  data_time: 0.4218  memory: 2602  loss: 0.6347  loss_cls: 0.2892  loss_bbox: 0.3454
    06/11 16:34:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][18/29]  lr: 2.2743e-03  eta: 0:00:09  time: 0.6168  data_time: 0.4237  memory: 2602  loss: 0.6347  loss_cls: 0.2898  loss_bbox: 0.3448
    06/11 16:34:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][19/29]  lr: 2.2783e-03  eta: 0:00:08  time: 0.6166  data_time: 0.4240  memory: 2602  loss: 0.6368  loss_cls: 0.2891  loss_bbox: 0.3477
    06/11 16:34:25 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][20/29]  lr: 2.2823e-03  eta: 0:00:07  time: 0.6175  data_time: 0.4242  memory: 2602  loss: 0.6361  loss_cls: 0.2880  loss_bbox: 0.3481
    06/11 16:34:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][21/29]  lr: 2.2863e-03  eta: 0:00:06  time: 0.6405  data_time: 0.4459  memory: 2602  loss: 0.6334  loss_cls: 0.2861  loss_bbox: 0.3474
    06/11 16:34:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][22/29]  lr: 2.2903e-03  eta: 0:00:05  time: 0.5934  data_time: 0.4000  memory: 2602  loss: 0.6290  loss_cls: 0.2826  loss_bbox: 0.3464
    06/11 16:34:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][23/29]  lr: 2.2943e-03  eta: 0:00:05  time: 0.5926  data_time: 0.3995  memory: 2602  loss: 0.6273  loss_cls: 0.2812  loss_bbox: 0.3462
    06/11 16:34:27 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][24/29]  lr: 2.2983e-03  eta: 0:00:04  time: 0.5922  data_time: 0.3991  memory: 2602  loss: 0.6255  loss_cls: 0.2810  loss_bbox: 0.3445
    06/11 16:34:29 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][25/29]  lr: 2.3023e-03  eta: 0:00:03  time: 0.6255  data_time: 0.4304  memory: 2602  loss: 0.6228  loss_cls: 0.2788  loss_bbox: 0.3440
    06/11 16:34:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][26/29]  lr: 2.3063e-03  eta: 0:00:02  time: 0.6028  data_time: 0.4066  memory: 2602  loss: 0.6195  loss_cls: 0.2773  loss_bbox: 0.3422
    06/11 16:34:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][27/29]  lr: 2.3103e-03  eta: 0:00:01  time: 0.5977  data_time: 0.4007  memory: 2602  loss: 0.6204  loss_cls: 0.2785  loss_bbox: 0.3419
    06/11 16:34:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][28/29]  lr: 2.3143e-03  eta: 0:00:00  time: 0.5973  data_time: 0.4003  memory: 2602  loss: 0.6195  loss_cls: 0.2792  loss_bbox: 0.3403
    06/11 16:34:30 - mmengine - [4m[97mINFO[0m - Exp name: rtmdet_tiny_drink_20230611_162224
    06/11 16:34:30 - mmengine - [4m[97mINFO[0m - Epoch(train) [20][29/29]  lr: 2.3183e-03  eta: 0:00:00  time: 0.5993  data_time: 0.4030  memory: 1345  loss: 0.6188  loss_cls: 0.2791  loss_bbox: 0.3397
    06/11 16:34:30 - mmengine - [4m[97mINFO[0m - Saving checkpoint at 20 epochs
    06/11 16:34:36 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 1/14]    eta: 0:00:20  time: 0.6680  data_time: 0.5691  memory: 245  
    06/11 16:34:36 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 2/14]    eta: 0:00:09  time: 0.6629  data_time: 0.5649  memory: 245  
    06/11 16:34:37 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 3/14]    eta: 0:00:10  time: 0.6672  data_time: 0.5687  memory: 245  
    06/11 16:34:38 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 4/14]    eta: 0:00:07  time: 0.6652  data_time: 0.5672  memory: 245  
    06/11 16:34:39 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 5/14]    eta: 0:00:07  time: 0.6705  data_time: 0.5716  memory: 245  
    06/11 16:34:39 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 6/14]    eta: 0:00:05  time: 0.6630  data_time: 0.5649  memory: 245  
    06/11 16:34:40 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 7/14]    eta: 0:00:05  time: 0.6707  data_time: 0.5728  memory: 245  
    06/11 16:34:40 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 8/14]    eta: 0:00:04  time: 0.6598  data_time: 0.5619  memory: 245  
    06/11 16:34:41 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][ 9/14]    eta: 0:00:03  time: 0.6557  data_time: 0.5586  memory: 245  
    06/11 16:34:41 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][10/14]    eta: 0:00:02  time: 0.6549  data_time: 0.5585  memory: 245  
    06/11 16:34:43 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][11/14]    eta: 0:00:02  time: 0.6535  data_time: 0.5581  memory: 245  
    06/11 16:34:43 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][12/14]    eta: 0:00:01  time: 0.6592  data_time: 0.5630  memory: 245  
    06/11 16:34:44 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][13/14]    eta: 0:00:00  time: 0.6503  data_time: 0.5551  memory: 245  
    06/11 16:34:44 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][14/14]    eta: 0:00:00  time: 0.6589  data_time: 0.5638  memory: 245  
    06/11 16:34:44 - mmengine - [4m[97mINFO[0m - Evaluating bbox...
    Loading and preparing results...
    DONE (t=0.02s)
    creating index...
    index created!
    Running per image evaluation...
    Evaluate annotation type *bbox*
    DONE (t=0.60s).
    Accumulating evaluation results...
    DONE (t=0.15s).
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.635
     Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.937
     Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.733
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.635
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.652
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.729
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.730
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
     Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.730
    06/11 16:34:45 - mmengine - [4m[97mINFO[0m - bbox_mAP_copypaste: 0.635 0.937 0.733 -1.000 -1.000 0.635
    06/11 16:34:45 - mmengine - [4m[97mINFO[0m - Epoch(val) [20][14/14]    coco/bbox_mAP: 0.6350  coco/bbox_mAP_50: 0.9370  coco/bbox_mAP_75: 0.7330  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.6350  data_time: 0.5607  time: 0.6496



```python

```
