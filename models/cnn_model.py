import torch.nn as nn
import torchvision.models as models

class LightweightModel(nn.Module):
    def __init__(self, num_classes=15):  # 固定输出15个类别
        super().__init__()
        self.backbone = models.mobilenet_v2(weights='DEFAULT')
        # 修改输入层适配4通道数据
        self.backbone.features[0][0] = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 修改分类层
        '''self.backbone.classifier[1] = nn.Linear(1280, num_classes)'''
        # 分类器回归头
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),  # 将原0.2改为0.5
            nn.Linear(1280, num_classes)
        )

        # 评估回归头
        # 改进角度回归头
        self.angle_regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.GELU(),
            nn.Linear(512, 3),
            nn.Tanh()  # 输出归一化到[-1,1]
        )

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        x = x.view(B * T, 4, x.shape[3], x.shape[4])
        features = self.backbone.features(x).mean([2, 3])

        # 分类输出
        cls_out = self.classifier(features).view(B, T, -1).mean(1)
        # 角度输出处理
        ang_out = self.angle_regressor(features) * 180  # 映射到[-180°,180°]
        ang_out = ang_out.view(B, T, -1).mean(1)

        return cls_out, ang_out


