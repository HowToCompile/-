# validate_data_split.py
import os
from pathlib import Path
from collections import defaultdict


def validate_data_splits(data_root="S:/myproject/data"):
    """
    验证数据划分完整性，不依赖预处理代码，仅检查最终存储路径

    参数：
        data_root: 预处理后数据根目录，包含 train/val/test 子目录
    """
    # 定义待检查的数据集划分
    splits = ['train', 'val', 'test']

    # 存储各集合的受试者-动作组合
    split_records = defaultdict(set)

    # 遍历所有数据文件
    for split in splits:
        split_dir = Path(data_root) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {split_dir}")

        # 收集当前集合的所有样本特征
        for file_path in split_dir.glob("*.npy"):
            # 解析文件名格式（示例: S01_A01.npy）
            stem = file_path.stem
            if "_" not in stem:
                print(f"⚠️ 文件名格式异常: {file_path}")
                continue

            subj, action = stem.split("_", 1)
            action_code = action.split(".")[0]  # 处理可能的多个扩展名分隔符

            # 记录关键特征组合（可扩展更多检查维度）
            identifier = (subj, action_code)
            split_records[split].add(identifier)

    # 执行多级数据验证
    validation_passed = True

    # 验证1: 检查跨集合的样本重复
    print("\n=== 跨集合样本重复检查 ===")
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            set_a = split_records[splits[i]]
            set_b = split_records[splits[j]]
            overlap = set_a & set_b

            if overlap:
                validation_passed = False
                print(f"❌ 发现 {splits[i]} 和 {splits[j]} 有 {len(overlap)} 个重复样本")
                print(f"    前5个重复样本示例: {list(overlap)[:5]}")
            else:
                print(f"✅ {splits[i]} 和 {splits[j]} 无样本重叠")

    # 验证2: 检查同一受试者在多个集合的情况（可选）
    print("\n=== 受试者分布检查 ===")
    subj_distribution = defaultdict(set)
    for split, records in split_records.items():
        for (subj, action) in records:
            subj_distribution[subj].add(split)

    problematic_subjs = []
    for subj, splits in subj_distribution.items():
        if len(splits) > 1:
            problematic_subjs.append((subj, splits))

    if problematic_subjs:
        validation_passed = False
        print(f"❌ 发现 {len(problematic_subjs)} 个受试者出现在多个集合")
        print(f"    示例受试者分布:")
        for subj, splits in problematic_subjs[:3]:
            print(f"    {subj}: {', '.join(splits)}")
    else:
        print("✅ 所有受试者仅出现在单一集合中")

    # 综合结果输出
    print("\n=== 最终验证结果 ===")
    if validation_passed:
        print("✅ 所有检查通过，数据划分有效")
    else:
        print("❌ 发现潜在数据泄露问题，请检查上述警告")
        raise ValueError("数据划分验证失败")


if __name__ == "__main__":
    validate_data_splits()