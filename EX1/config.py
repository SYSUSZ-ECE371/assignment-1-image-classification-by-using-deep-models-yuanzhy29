_base_ = [
    'mmpretrain/configs/_base_/models/resnet152.py',         
    'mmpretrain/configs/_base_/datasets/imagenet_bs64.py',    
    'mmpretrain/configs/_base_/schedules/imagenet_bs256.py', 
    'mmpretrain/configs/_base_/default_runtime.py',           
]

# 模型设置
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.pytorch.org/models/resnet152-b121ed2d.pth'
        )
    ),
    head=dict(  # 分类头配置
        num_classes=5,  # 分类类别数改为5类
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),  # 设置损失函数为交叉熵损失
        topk=(1,),  # 设置评估指标为top1准确率
    )
)

# 数据集配置
dataset_type = 'ImageNet'  # 自定义数据集类型
data_preprocessor = dict(
    num_classes=5,  # 数据预处理器的类别数设置为5类
)

train_pipeline = [
    dict(type='LoadImageFromFile'),  # 加载图像
    dict(type='RandomResizedCrop', scale=224),  # 随机剪裁并调整大小到224x224
    dict(type='RandomFlip', prob=0.5),  # 随机水平翻转
    dict(type='PackInputs'),  # 打包输入数据
]

test_pipeline = [
    dict(type='LoadImageFromFile'),  # 加载图像
    dict(type='Resize', scale=256),  # 调整大小到256x256
    dict(type='CenterCrop', crop_size=224),  # 中心剪裁到224x224
    dict(type='PackInputs'),  # 打包输入数据
]

train_dataloader = dict(
    batch_size=32,  # 训练时的批次大小
    dataset=dict(
        type=dataset_type,
        data_root='../data',  # 数据集根目录路径
        ann_file='train.txt',  # 训练集标注文件
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,  # 验证时的批次大小
    dataset=dict(
        type=dataset_type,
        data_root='../data',  # 数据集根目录路径
        ann_file='val.txt',  # 验证集标注文件
        pipeline=test_pipeline
    ),
)

test_dataloader = val_dataloader  # 测试数据加载器与验证数据加载器配置相同

# 优化器配置
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',  # 使用随机梯度下降优化器
        lr=0.01,  # 初始学习率
        momentum=0.9,  # 动量
        weight_decay=0.0001  # 权重衰减
    ),
)

# 学习率调度器配置
param_scheduler = [
    dict(
        type='MultiStepLR',  # 多步学习率调度器
        milestones=[30, 60, 90],  # 在第30、60、90个epoch时降低学习率
        gamma=0.1  # 学习率衰减系数
    )
]

# 训练相关配置
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)  # 设置训练总epoch数为100，每个epoch验证一次

# 评估相关配置
val_evaluator = dict(type='Accuracy', topk=1)  # 验证时的评估指标为top1准确率
test_evaluator = val_evaluator  # 测试时的评估指标与验证时相同

# 日志和保存配置
default_hooks = dict(
    checkpoint=dict(interval=10),  # 每10个epoch保存一次模型
    logger=dict(interval=100)  # 每100个迭代打印一次日志
)