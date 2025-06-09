import torch
import numpy as np
from net_model import ICNET

# 参数设置
m2 = 100
samplenum = 2 * m2  # 测试集样本总数
batch_size = 2  # 每轮的样本大小
w_HR = 128  # 样本尺寸

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 载入网络与测试集
net = ICNET().to(device)  # 实例化模型并移至设备
state_dict = torch.load('net_3.pkl', map_location=device)  # 加载模型状态字典
net.load_state_dict(state_dict)  # 将状态字典加载到模型中
net.eval()  # 设置模型为评估模式

# 加载测试数据
x = np.load('test_set.npy') / 255  # 归一化
x = torch.tensor(x).float().to(device)
y = torch.cat((torch.zeros(m2), torch.ones(m2))).long().to(device)  # 标签

# 初始化变量
correct = 0

# 测试
with torch.no_grad():  # 禁用梯度计算
    for batch_idx in range(0, samplenum, batch_size):
        # 切片获取当前批次数据
        x_batch = x[batch_idx:batch_idx + batch_size]
        y_batch = y[batch_idx:batch_idx + batch_size]

        # 前向传播
        outputs = net(x_batch)
        _, predicted = torch.max(outputs.data, 1)

        # 计算准确率
        correct += (predicted == y_batch).sum().item()

# 打印准确率
accuracy = 100 * correct / samplenum
print(f'Accuracy of the network on the {samplenum} test images: {accuracy:.2f}%')