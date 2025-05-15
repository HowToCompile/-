import os
import numpy as np
from torch.utils.data import Dataset
import torch


class MMFiDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.gt_dir = os.path.join(data_dir, f"gt_{split}")
        self.samples = []

        # 新增字段：解析文件名中的S和A信息
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".npy"):
                parts = fname.split("_")
                subj = parts[0][1:]  # 提取S后的数字，如"S01"→"01"
                action = parts[1].split(".")[0]
                self.samples.append({
                    'file': fname,
                    'subj': subj,
                    'action': action
                })

        self.label_map = {
            'A01': 0, 'A02': 1, 'A03': 2, 'A04': 3,
            'A08': 4, 'A09': 5, 'A11': 6, 'A14': 7,
            'A15': 8, 'A18': 9, 'A19': 10, 'A20': 11,
            'A23': 12, 'A24': 13, 'A27': 14
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 加载数据 (保持原有逻辑)
        data = np.load(os.path.join(self.data_dir, sample['file']))
        # 加载GT并计算角度
        subj_action = sample['file'].split(".")[0]
        gt = np.load(os.path.join(self.gt_dir, f"{subj_action}.npy"))
        angles = self._calc_joint_angles(gt)

        return {
            'data': torch.FloatTensor(data),
            'label': torch.LongTensor([self.label_map[sample['action']]]),
            'subj': sample['subj'],  # 新增字段
            'action': sample['action'],  # 新增字段
            'angles': torch.FloatTensor(angles)
        }

    # 保持原有角度计算和工具方法不变

    def _calc_joint_angles(self, gt):
        """计算三个关键角度"""
        angles = []
        for t in range(gt.shape[0]):
            # 右肘弯曲角（关节5-6-7）
            r_shoulder = gt[t, 5]
            r_elbow = gt[t, 6]
            r_wrist = gt[t, 7]
            angle1 = self._vector_angle(r_elbow - r_shoulder, r_wrist - r_elbow)

            # 左膝弯曲角（关节12-13-14）
            l_hip = gt[t, 12]
            l_knee = gt[t, 13]
            l_ankle = gt[t, 14]
            angle2 = self._vector_angle(l_knee - l_hip, l_ankle - l_knee)

            # 躯干倾斜角（关节5-8到12-15的中线）
            shoulder_center = (gt[t, 5] + gt[t, 8]) / 2
            hip_center = (gt[t, 12] + gt[t, 15]) / 2
            angle3 = self._vector_angle(hip_center - shoulder_center, [0, 1, 0])  # 与垂直方向夹角

            angles.append([angle1, angle2, angle3])
        return np.array(angles).mean(axis=0)  # 时间平均

    # utils/dataset.py 修改点
    def _vector_angle(self, v1, v2):
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos = np.clip(cos, -1.0, 1.0)  # 新增数值裁剪
        return np.degrees(np.arccos(cos))

    def get_angle_names(self):
        return ["右肘弯曲角", "左膝弯曲角", "躯干倾斜度"]

    def get_action_name(self, label):
        return [k for k, v in self.label_map.items() if v == label][0]
'''class MMFiDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.gt_dir = os.path.join(data_dir, f"gt_{split}")  # GT目录
        self.samples = []

        # 建立数据与GT的映射
        for fname in os.listdir(self.data_dir):
            if fname.endswith(".npy"):
                subj_action = fname.split(".")[0]
                gt_path = os.path.join(self.gt_dir, f"{subj_action}.npy")
                if os.path.exists(gt_path):
                    self.samples.append({
                        "data": os.path.join(self.data_dir, fname),
                        "gt": gt_path,
                        "action": subj_action.split("_")[1]
                    })
        # 固定标签映射表
        self.label_map = {
            'A01': 0, 'A02': 1, 'A03': 2, 'A04': 3,
            'A08': 4, 'A09': 5, 'A11': 6, 'A14': 7,
            'A15': 8, 'A18': 9, 'A19': 10, 'A20': 11,
            'A23': 12, 'A24': 13, 'A27': 14
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载融合数据
        data = np.load(sample["data"])  # [T,4,H,W]

        # 加载并处理GT
        gt = np.load(sample["gt"])
        angles = self._process_gt(gt)  # [num_angles]

        return {
            'data': torch.FloatTensor(data),
            'label': torch.LongTensor([self.label_map[sample["action"]]]),
            'angles': torch.FloatTensor(angles)
        }

    def _process_gt(self, gt_data):
        """计算5个关键关节角度（示例）"""
        # 假设gt_data形状为[T, J, 3]
        angles = []
        for t in range(gt_data.shape[0]):
            # 右肘角度计算
            r_shoulder = gt_data[t, 5]  # 关节索引需根据实际数据调整
            r_elbow = gt_data[t, 6]
            r_wrist = gt_data[t, 7]
            angle1 = self._calc_angle(r_shoulder, r_elbow, r_wrist)

            # 左膝角度计算
            l_hip = gt_data[t, 12]
            l_knee = gt_data[t, 13]
            l_ankle = gt_data[t, 14]
            angle2 = self._calc_angle(l_hip, l_knee, l_ankle)

            angles.append([angle1, angle2])
        return np.array(angles).mean(axis=0)  # 时间维度平均

    def _calc_angle(self, a, b, c):
        """三点计算关节角度"""
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return np.degrees(np.arccos(cosine_angle))'''''