import json
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from models.cnn_model import LightweightModel
from utils.dataset import MMFiDataset
# 关键修复：添加matplotlib导入
import matplotlib.pyplot as plt

def generate_advice(sample, label_map):
    """生成个性化训练建议"""
    advice = []

    # 关节角度建议规则
    joint_rules = {
        "右肘弯曲角": [
            (5, "肘部稳定性良好"),
            (10, "注意肘关节活动幅度，避免过伸"),
            (float('inf'), "建议使用护具并咨询康复师")
        ],
        "左膝弯曲角": [
            (8, "膝关节活动范围正常"),
            (15, "加强股四头肌力量训练"),
            (float('inf'), "存在受伤风险，建议医疗评估")
        ],
        "躯干倾斜度": [
            (5, "躯干控制良好"),
            (10, "注意核心肌群发力模式"),
            (float('inf'), "存在脊柱代偿，需调整姿势")
        ]
    }

    # 处理每个关节的角度误差
    for joint, err_str in sample["角度误差"].items():
        err = float(err_str.strip('°'))
        for threshold, msg in joint_rules.get(joint, []):
            if err <= threshold:
                advice.append(f"{joint} ({err}°): {msg}")
                break

    # 处理分类错误
    if sample["预测标签"] != sample["真实标签"]:
        try:
            pred_action = [k for k, v in label_map.items() if v == sample["预测标签"]][0]
            true_action = [k for k, v in label_map.items() if v == sample["真实标签"]][0]
            advice.append(f"动作识别错误: 将{true_action}误判为{pred_action}, 请检查动作规范性")
        except:
            advice.append("动作识别错误: 标签映射异常")

    return advice

def evaluate_random_samples(model, loader, device, num_samples=30):
    """
    随机评估指定数量的样本（默认30个）
    返回格式：{
        "总体统计": {平均准确率, 平均角度误差},
        "详细结果": [含个性化建议的样本数据]
    }
    """
    model.eval()
    stats = {
        'total': 0,
        'cls_correct': 0,
        'angle_errors': defaultdict(float),
        'samples': []
    }

    # 获取数据集元信息
    raw_dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset
    angle_names = raw_dataset.get_angle_names()
    label_map = raw_dataset.label_map

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估进度", unit="batch"):
            inputs = batch['data'].to(device)
            labels = batch['label'].squeeze().to(device)
            angles = batch['angles'].to(device)
            subjs = batch['subj']
            actions = batch['action']

            # 获取预测结果
            cls_pred, ang_pred = model(inputs)

            # 处理批次数据
            batch_size = labels.size(0)
            for i in range(batch_size):
                # 基础统计
                stats['total'] += 1
                correct = (cls_pred[i].argmax() == labels[i]).item()
                stats['cls_correct'] += correct

                # 角度误差计算
                sample_errors = torch.abs(ang_pred[i] - angles[i])
                angle_dict = {name: f"{sample_errors[j].item():.1f}°"
                              for j, name in enumerate(angle_names)}

                # 构建样本数据
                sample_data = {
                    "受试者": subjs[i],
                    "动作": actions[i],
                    "预测标签": cls_pred[i].argmax().item(),
                    "真实标签": labels[i].item(),
                    "角度误差": angle_dict,
                    "个性化建议": generate_advice({
                        "预测标签": cls_pred[i].argmax().item(),
                        "真实标签": labels[i].item(),
                        "角度误差": angle_dict
                    }, label_map)
                }
                stats['samples'].append(sample_data)

                # 累计角度误差
                for j, name in enumerate(angle_names):
                    stats['angle_errors'][name] += sample_errors[j].item()

                # 提前终止
                if stats['total'] >= num_samples:
                    break
            if stats['total'] >= num_samples:
                break

    # 生成报告
    report = {
        "评估设置": {
            "总样本数": stats['total'],
            "随机种子": random.getstate()[1][0],
            "评估时间": torch.cuda.get_device_properties(device).name if device.type == 'cuda' else "CPU"
        },
        "总体统计": {
            "分类准确率": f"{stats['cls_correct'] / stats['total'] * 100:.1f}%",
            "平均角度误差": {
                name: f"{stats['angle_errors'][name] / stats['total']:.1f}°"
                for name in angle_names
            }
        },
        "详细结果": stats['samples'][:num_samples]
    }

    # 质量评估
    max_error = max(float(v.strip('°')) for v in report["总体统计"]["平均角度误差"].values())
    report["系统建议"] = ("各指标均在安全范围内" if max_error < 5 else
                          "存在需要注意的动作模式" if max_error < 10 else
                          "检测到高风险动作模式，建议详细评估")

    # 新增关节误差可视化（关键修复）
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False

    avg_errors = report["总体统计"]["平均角度误差"]
    joints = list(avg_errors.keys())
    values = [float(v.strip('°')) for v in avg_errors.values()]

    plt.figure(figsize=(10, 5))
    plt.barh(joints, values, color='#2c7fb8')
    plt.title('关节角度平均误差', fontsize=14)
    plt.xlabel('误差 (°)', fontsize=12)
    plt.xlim(0, max(values) * 1.2)
    plt.grid(axis='x', linestyle='--')

    for i, v in enumerate(values):
        plt.text(v + 0.5, i, f"{v:.1f}°", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('joint_errors.png', dpi=300, bbox_inches='tight')
    plt.close()

    return report

if __name__ == "__main__":
    # 初始化配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = LightweightModel().to(device)
    model.load_state_dict(torch.load(
        "S:/myproject/best_model.pth",
        map_location=device,
        weights_only=True
    ))

    # 准备数据
    full_dataset = MMFiDataset("S:/myproject/data", "test")
    eval_indices = random.sample(range(len(full_dataset)), min(30, len(full_dataset)))
    eval_loader = DataLoader(
        Subset(full_dataset, eval_indices),
        batch_size=8,
        shuffle=False,
        num_workers=2
    )

    # 执行评估
    report = evaluate_random_samples(model, eval_loader, device)

    # 输出结果
    print("\n智能评估报告：")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 保存报告
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"assessment_report_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)