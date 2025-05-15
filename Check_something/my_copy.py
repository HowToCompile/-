import torch
from torch.utils.data import DataLoader
from models.cnn_model import LightweightModel
from utils.dataset import MMFiDataset


def main():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型初始化
    model = LightweightModel(num_classes=15).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-3,
        weight_decay=1e-4  # 改动点1：添加L2正则化
    )
    criterion = torch.nn.CrossEntropyLoss()

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # 改动点2
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # 早停机制初始化
    best_val_acc = 0.0
    early_stop_patience = 3
    no_improve_count = 0  # 改动点3

    # 数据加载配置
    train_dataset = MMFiDataset("/data", split="train")
    val_dataset = MMFiDataset("/data", split="val")
    test_dataset = MMFiDataset("/data", split="test")

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # 训练循环
    for epoch in range(20):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播 + 梯度裁剪
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 改动点4
            optimizer.step()

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.squeeze().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_loss += loss.item()

        # 计算指标
        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100

        # 学习率调整
        scheduler.step(val_acc)  # 改动点2延续

        # 早停判断
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save(model.state_dict(), "../best_model.pth")
        else:
            no_improve_count += 1
            if no_improve_count >= early_stop_patience:
                print(f"早停触发于Epoch {epoch}")
                break  # 终止训练

        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss / len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)

    # ====== 最终测试 ======
    model.load_state_dict(torch.load("../best_model.pth"))  # 加载最佳模型
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print("\nFinal Test Accuracy: {:.2f}%".format(test_correct / test_total * 100))


if __name__ == '__main__':
    main()








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
        # 若分类器中有Dropout层（如原代码未包含需添加）
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),  # 将原0.2改为0.5
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        x = x.view(B * T, 4, x.shape[3], x.shape[4])  # [B*T,4,H,W]
        x = self.backbone(x)  # [B*T, num_classes]
        return x.view(B, T, -1).mean(dim=1)  # [B, num_classes]