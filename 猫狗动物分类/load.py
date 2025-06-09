import numpy as np
import cv2

def load_images(m, start_idx, lenth, folder_path):
    """
    加载图片并返回一个 NumPy 数组
    :param m: 图片数量
    :param start_idx: 起始索引
    :param lenth: 图片大小
    :param folder_path: 图片文件夹路径
    :return: NumPy 数组
    """
    dataset = np.zeros((m * 2, 3, lenth, lenth), dtype=np.float32)
    success_count = 0

    for i in range(m):
        paths = [
            f'{folder_path}/cats/cat.{start_idx + i + 1}.jpg',
            f'{folder_path}/dogs/dog.{start_idx + i + 1}.jpg'
        ]
        for idx, path in enumerate(paths):
            img = cv2.imread(path)
            if img is None:
                print(f"无法读取图像: {path}")
                continue
            img = cv2.resize(img, (lenth, lenth))
            dataset[success_count, 0, :, :] = img[:, :, 0]
            dataset[success_count, 1, :, :] = img[:, :, 1]
            dataset[success_count, 2, :, :] = img[:, :, 2]
            success_count += 1

    return dataset[:success_count]  # 截取实际加载的图片

# 参数设置
m1 = 700  # 训练集数量
m2 = 100  # 测试集数量
lenth = 128  # 图片大小
train_folder = './sample_3'
test_folder = './sample_3'

# 加载数据
train_set = load_images(m1, 0, lenth, train_folder)
test_set = load_images(m2, m1, lenth, test_folder)

# 保存为 .npy 文件
np.save('train_set.npy', train_set)
np.save('test_set.npy', test_set)
print("数据保存成功！")