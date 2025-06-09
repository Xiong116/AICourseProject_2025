##用于测试test_set.npy train_set.npy是否正确写入
import numpy as np
import matplotlib.pyplot as plt

#要测试的下标
x1 = 69
x2 = 59

#读取测试集与训练集
train_set = np.load('train_set.npy')
test_set = np.load('test_set.npy')

#选择对应图片
img1 = train_set[x1,0,:,:]
img2 = test_set[x2,0,:,:]
img1 = img1.astype(np.uint8)
img2 = img2.astype(np.uint8)

#显示图片
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img1, cmap='gray')
plt.title(f'Image {x1} (Train Set)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img2, cmap='gray')
plt.title(f'Image {x2} (Test Set)')
plt.axis('off')

plt.show()

