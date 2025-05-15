import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ACTION_MAP = {
    'A01': 0, 'A02': 1, 'A03': 2, 'A04': 3,
    'A08': 4, 'A09': 5, 'A11': 6, 'A14': 7,
    'A15': 8, 'A18': 9, 'A19': 10, 'A20': 11,
    'A23': 12, 'A24': 13, 'A27': 14
}


def correlated_illumination(rgb, depth):
    """光照-深度协同增强 (单帧处理)"""
    # 亮度调整幅度与深度误差关联
    brightness_factor = np.random.uniform(0.85, 1.15)
    depth_error = 0.95 + 0.1 * (brightness_factor - 1)  # 亮度↑则深度↓

    # 应用基础变换
    augmented_rgb = np.clip(rgb * brightness_factor, 0.0, 1.0)
    augmented_depth = np.clip(depth * depth_error, 0.0, 1.0)

    # 添加传感器噪声 (RGB噪声强于Depth)
    rgb_noise = np.random.normal(0, 0.02, augmented_rgb.shape).astype(np.float16)
    depth_noise = np.random.normal(0, 0.01, augmented_depth.shape).astype(np.float16)

    return (
        np.clip(augmented_rgb + rgb_noise, 0.0, 1.0),
        np.clip(augmented_depth + depth_noise, 0.0, 1.0)
    )


def preprocess_rgb_depth(rgb_dir, depth_dir, target_size=(112, 112), max_frames=30, augment=False):
    """新增augment参数控制增强"""
    try:
        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith('.npy')])[:max_frames]
        depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png')])[:max_frames]
        min_frames = min(len(rgb_files), len(depth_files))
        if min_frames == 0:
            raise ValueError(f"Empty directory: {rgb_dir} or {depth_dir}")

        fused = []
        for rgb_file, depth_file in zip(rgb_files[:min_frames], depth_files[:min_frames]):
            # 处理RGB
            rgb = np.load(os.path.join(rgb_dir, rgb_file))
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            elif rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            rgb = cv2.resize(rgb, target_size).astype(np.float32) / 255.0
            rgb = np.transpose(rgb, (2, 0, 1))  # [3,H,W]

            # 处理Depth
            depth = cv2.imread(os.path.join(depth_dir, depth_file), cv2.IMREAD_GRAYSCALE)
            depth = cv2.resize(depth, target_size).astype(np.float32) / 65535.0
            depth = np.expand_dims(depth, axis=0)  # [1,H,W]

            # 应用增强（仅训练集且50%概率）
            if augment and np.random.rand() < 0.5:
                rgb, depth = correlated_illumination(rgb, depth)

            fused.append(np.concatenate([rgb, depth], axis=0))

        return np.array(fused, dtype=np.float16)  # [T,4,H,W]
    except Exception as e:
        print(f"Error in {rgb_dir}: {str(e)}")
        return None


def preprocess_and_split(data_root="S:/myproject/MMFi", output_dir="S:/myproject/data"):
    samples = []
    # ... [保持原有数据收集逻辑不变] ...
    # 遍历数据集，仅处理已知的15个动作
    for subj in os.listdir(data_root):
        subj_dir = os.path.join(data_root, subj)
        if not os.path.isdir(subj_dir):
            continue
        for action in os.listdir(subj_dir):
            if action not in ACTION_MAP:
                continue  # 跳过未知动作
            action_dir = os.path.join(subj_dir, action)
            gt_path = os.path.join(action_dir, "ground_truth.npy")  # ← 新增
            rgb_dir = os.path.join(action_dir, "rgb")
            depth_dir = os.path.join(action_dir, "depth")
            if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
                samples.append({
                    "subj": subj,
                    "action": action,
                    "rgb_dir": rgb_dir,
                    "depth_dir": depth_dir,
                    "gt_path": gt_path  # ← 关键新增

                })

    # 划分数据集
    train, test = train_test_split(samples, test_size=0.4, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)


    # 处理并保存（新增增强控制）
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        split_dir = os.path.join(output_dir, split_name)
        gt_dir = os.path.join(output_dir, f"gt_{split_name}")  # GT存储目录
        os.makedirs(split_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        for sample in split_data:
            output_path = os.path.join(split_dir, f"{sample['subj']}_{sample['action']}.npy")
            if os.path.exists(output_path):
                print(f"Skipped (exists): {output_path}")
                continue


            # 仅在训练时启用增强
            data = preprocess_rgb_depth(
                sample["rgb_dir"],
                sample["depth_dir"],
                augment=(split_name == "train")
            )

            if data is not None:
                np.save(output_path, data)
                print(f"Saved: {output_path}")

                # 新增GT保存
            gt_data = np.load(sample["gt_path"])
            gt_output_path = os.path.join(gt_dir, f"{sample['subj']}_{sample['action']}.npy")
            np.save(gt_output_path, gt_data)


if __name__ == "__main__":
    preprocess_and_split()