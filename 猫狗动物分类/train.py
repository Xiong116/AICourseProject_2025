import torch
import torch.nn as nn
import numpy as np
from net_model import ICNET

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置
m1 = 700
samplenum = 2 * m1  # 训练集样本总数
batch_size = 2  # 每轮的样本大小
epochs = 100
lr = 0.001
weight_decay = 0.0001

# 加载数据
x = (torch.tensor(np.load('train_set.npy')) / 255).float().to(device)
y = torch.cat((torch.zeros(samplenum // 2), torch.ones(samplenum // 2))).long().to(device)

# 初始化模型、优化器和损失函数
model = ICNET().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss_func = nn.CrossEntropyLoss()

# 训练
for epoch in range(epochs):
    for batch_idx in range(0, samplenum, batch_size):
        # 切片获取当前批次数据
        x_batch = x[batch_idx:batch_idx + batch_size]
        y_batch = y[batch_idx:batch_idx + batch_size]

        # 前向传播
        output = model(x_batch)
        loss = loss_func(output, y_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印日志
        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{samplenum}], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'net_3.pkl')
print("模型保存成功！")