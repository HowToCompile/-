import numpy as np
import scipy

# 加载示例文件
gt = np.load("/MMFi/S01/A01/ground_truth.npy", allow_pickle=True)
rgb1 = np.load("/MMFi/S01/A01/rgb/frame001.npy", allow_pickle=True)
infra_1 = np.load("/MMFi/S01/A01/infra1/frame001.npy", allow_pickle=True)
#data_shape = np.load("S:/myproject/data/train/S01_A02.npy", allow_pickle=True)

print(gt)
#print("rgb")
#print(rgb1)
#print("infra1")
#print(infra_1)
#print(data_shape)
# 典型数据结构（根据MMFi论文推测）：
'''{
    'action_label': 'side_stretch',  # 动作类别名称
    'timestamps': [
        [0.0, 2.5],   # 动作开始-结束时间（秒）
        [3.0, 5.5]    # 多段动作的时间戳
    ],
    'quality_scores': {
        'range_of_motion': 0.85,    # 关节活动范围评分（0-1）
        'posture_accuracy': 0.72,   # 姿势准确度
        'expert_rating': 4.3       # 专家综合评分（1-5分）
    },
    'keypoints': np.array([...])   # 关节点坐标（若有）
}'''