import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from models.cnn_model import LightweightModel
from utils.dataset import MMFiDataset
# 新增可视化相关导入
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 启用 CuDNN 自动优化
torch.backends.cudnn.benchmark = True


def main():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = LightweightModel(num_classes=15).to(device)

    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=5e-4
    )

    # 损失函数
    criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.02)
    criterion_ang = nn.L1Loss()

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # 数据加载
    train_dataset = MMFiDataset("S:/myproject/data", split="train")
    val_dataset = MMFiDataset("S:/myproject/data", split="val")
    test_dataset = MMFiDataset("S:/myproject/data", split="test")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # 混合精度训练
    scaler = GradScaler()
    best_val_acc = 0.0
    early_stop_counter = 0

    # 新增训练记录器
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'ang_mae': []
    }

    # 训练循环
    for epoch in range(20):
        # ========== 训练阶段 ==========
        model.train()
        train_cls_loss = 0.0
        train_correct = 0
        train_total = 0
        train_ang_loss = 0.0
        train_ang_error = 0.0

        for batch in train_loader:
            inputs = batch['data'].to(device)
            labels = batch['label'].squeeze().to(device)
            angles = batch['angles'].to(device)

            optimizer.zero_grad()

            # 前向传播
            with autocast():
                cls_pred, ang_pred = model(inputs)
                loss_cls = criterion_cls(cls_pred, labels)
                loss_ang = criterion_ang(ang_pred, angles)
                total_loss = loss_cls + 0.6 * loss_ang  # 原为0.3

            # 反向传播
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()

            # 统计指标
            train_cls_loss += loss_cls.item()
            train_ang_loss += loss_ang.item()
            train_ang_error += torch.abs(ang_pred - angles).mean().item()
            train_correct += (cls_pred.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        # ========== 验证阶段 ==========
        model.eval()
        val_cls_loss = 0.0
        val_correct = 0
        val_total = 0
        val_ang_loss = 0.0
        val_ang_error = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['data'].to(device)
                labels = batch['label'].squeeze().to(device)
                angles = batch['angles'].to(device)

                cls_pred, ang_pred = model(inputs)

                val_cls_loss += criterion_cls(cls_pred, labels).item()
                val_ang_loss += criterion_ang(ang_pred, angles).item()
                val_ang_error += torch.abs(ang_pred - angles).mean().item()
                val_correct += (cls_pred.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        # 计算统计量
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total

        # 记录历史数据（新增关键点）
        history['train_loss'].append(train_cls_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_cls_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['ang_mae'].append(val_ang_error / len(val_loader))

        # 更新学习率
        scheduler.step()

        # ========== 日志输出 ==========
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_cls_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_cls_loss / len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        print("\n[关节评估]")
        print(
            f"Train ANG Loss: {train_ang_loss / len(train_loader):.4f} | MAE: {train_ang_error / len(train_loader):.2f}°")
        print(f"Val ANG Loss: {val_ang_loss / len(val_loader):.4f} | MAE: {val_ang_error / len(val_loader):.2f}°")
        print("-" * 50)

        # 早停机制
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ========== 最终测试 ==========
    model.load_state_dict(torch.load(
        "best_model.pth",
        map_location=device,
        weights_only=True  # 修复安全警告
    ))
    model.eval()
    test_correct = 0
    test_total = 0
    test_ang_error = 0.0

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['data'].to(device)
            labels = batch['label'].squeeze().to(device)
            angles = batch['angles'].to(device)

            cls_pred, ang_pred = model(inputs)
            test_correct += (cls_pred.argmax(1) == labels).sum().item()
            test_total += labels.size(0)
            test_ang_error += torch.abs(ang_pred - angles).mean().item()

    print("\nFinal Test Results:")
    print(f"Test Accuracy: {100. * test_correct / test_total:.2f}%")
    print(f"Test ANG MAE: {test_ang_error / len(test_loader):.2f}°")

    # ========== 可视化部分 ==========
    # 1. 训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 混淆矩阵
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['data'].to(device)
            labels = batch['label'].squeeze().to(device)
            outputs, _ = model(inputs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(1).cpu().numpy())

    # 生成连续动作标签（关键修正）
    action_names = [f"A{(i + 1):02d}" for i in range(15)]
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=action_names,
                yticklabels=action_names)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()