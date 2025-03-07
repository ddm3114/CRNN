# RPN with FPN and CRNN

在Kaggle上收集了约800张外国车牌图像，通过线性插值将图片大小调整为480*480*3。利用Faster R-CNN进行两阶段目标检测，
先判断正负样本，受文字检测启发。采用RPN with FPN方法从特征图中多尺度提取信息，使用ResNet50作为特征提取骨干网
络。并通过CRNN对rol-pooling结果进行识别。
· 对ResNet的256、512、1024层进行了特征提取，得到不同尺寸的图像，分别为120、60、30。使用FPN进行多尺度融合，利用
1*1卷积实现上采样和通道数调整。RPN输出结果作为头部网络，通过卷积和全局平均池化得到预测框坐标和类别判断通道。
Anchor的选择通过滑动窗口和IOU阈值来确定正负样本。
· 采用CRNN模型进行字符识别，裁剪车牌内部部分并resize为100*32输入CRNN。利用Kaggle开源数据集进行训练，使用CTC
loss作为误差函数进行误差回传。尝试多个优化器，如Adam和SGD，使用预训练参数进行微调以降低损失，采用权重衰减和学习
率衰减策略。
