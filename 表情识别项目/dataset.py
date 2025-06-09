import os
import pandas as pd
from PIL import Image
import numpy as np
import utils

config = utils.read_config()
# 数据集路径
TRAIN_DIR = config['train_data_root']
TEST_DIR = config['test_data_root']
TRAIN_LABELS_FILE = config['train_label_root']
TEST_LABELS_FILE = config['test_label_root']

# 加载标签文件
def load_labels(label_file):
    df = pd.read_csv(label_file)
    labels = {}
    for _, row in df.iterrows():
        img_name, label = row["image"], int(row["label"]) - 1  # 标签从0开始
        labels[img_name] = label
    return labels

# 加载图像和标签
def load_dataset(data_dir, labels):
    images = []
    targets = []
    for img_name, label in labels.items():
        # 假设图片存储在对应类别的子文件夹中
        class_folder = str(label + 1)  # 类别从1开始
        img_path = os.path.join(data_dir, class_folder, img_name)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")  # 确保图像是RGB格式
            img = img.resize((224, 224))  # 调整大小以适应模型输入
            images.append(np.array(img))
            targets.append(label)
    return np.array(images), np.array(targets)

# 主函数
def main():
    # 加载训练集和测试集的标签
    train_labels = load_labels(TRAIN_LABELS_FILE)
    test_labels = load_labels(TEST_LABELS_FILE)

    # 加载训练集和测试集
    train_images, train_targets = load_dataset(TRAIN_DIR, train_labels)
    test_images, test_targets = load_dataset(TEST_DIR, test_labels)

    # 限制数据集大小
    max_train_size = config['max_train_size']
    max_test_size = config['max_test_size']

    if len(train_images) > max_train_size:
        train_images = train_images[:max_train_size]
        train_targets = train_targets[:max_train_size]

    if len(test_images) > max_test_size:
        test_images = test_images[:max_test_size]
        test_targets = test_targets[:max_test_size]

    print(f"调整后训练集大小: {len(train_images)}")
    print(f"调整后测试集大小: {len(test_images)}")
    print(f"调整后训练集标签大小: {len(train_targets)}")
    print(f"调整后测试集标签大小: {len(test_targets)}")

    train_set_path = os.path.join(config["save_path"],'train_set.npy' )
    test_set_path = os.path.join(config["save_path"], 'test_set.npy')
    train_labels_path = os.path.join(config['save_path'],'train_labels.npy')
    test_labels_path = os.path.join(config['save_path'], 'test_labels.npy')

    # 保存为 .npy 文件
    np.save(train_set_path, train_images)
    np.save(test_set_path, test_images)
    np.save(train_labels_path, train_targets)
    np.save(test_labels_path, test_targets)
    print("数据保存成功！")

if __name__ == "__main__":
    main()