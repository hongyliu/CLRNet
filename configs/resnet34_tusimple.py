net = dict(
    type='Detector',
)

backbone = dict(
    type='ResNetWrapper',
    resnet='resnet34',
    pretrained=True,
    replace_stride_with_dilation=[False, False, False],
    out_conv=False,
    in_channels=[64, 128, 256, 512]
)


aggregator = dict()

neck=dict(
    type='FPN',
    in_channels=[128, 256, 512],
    out_channels=64,
    num_outs=3,
)

heads=dict(
    type='ROIGather',
    in_channels=[64, 64, 64],
    feature_size=[(10, 25), (20, 50), (40, 100)],
    feature_channel=64,
    sample_points=36,
    resize_shape=(10, 25)
)

cost_weight = dict(
    cls_weight=1,
    sim_weight=3
)

loss_weights=dict(
    cls_weight=1,
    xytl_weight=1.,
    liou_weight=1.
)


batch_size = 2
prior_elements = 78
num_points = 72
num_sampled_points = 36
max_lanes = 5

optimizer = dict(type='Adam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

epochs = 70
total_iter = (3626 // batch_size) * epochs

scheduler = dict(
    type = 'MultiStepLR',
    milestones=[8, 14],
    gamma=0.1
)

seg_loss_weight = 1.0
eval_ep = 2
save_ep = 2 

img_norm = dict(
    mean=[75.3, 76.6, 77.6],
    std=[50.5, 53.8, 54.3]
)


cut_height = 0 
ori_img_h = 720
ori_img_w = 1280
img_w = 800
img_h = 320


train_process = [
    dict(type='GenerateLaneLine',
        transforms = (
            dict(
                name = 'Affine',
                parameters = dict(
                    translate_px = dict(
                        x=(-25, 25),
                        y = (-10, 10)
                    ),
                    rotate=(-6, 6),
                    scale=(0.85, 1.15)
                )
            ),
            dict(
                name='HorizontalFlip',
                parameters=dict(
                    p=0.5
                ),
            )
        ),
        wh = (img_w, img_h),
    ),
    dict(type='ToTensor', keys=['img', 'lane_prior']),
]

val_process = [
    dict(type='GenerateLaneLine', wh=(img_w, img_h)),
    dict(type='ToTensor', keys=['img']),
]


dataset_path = 'D:\\PyCharm\\WorkShop\\data\\tusimple'
test_json_file = 'D:\\PyCharm\\WorkShop\\data\\tusimple\\test_label.json'
dataset_type = 'TuSimple'

dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='trainval',
        processes=train_process,
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        processes=val_process,
    )
)


workers = 0
log_interval = 1000
seed = 42
