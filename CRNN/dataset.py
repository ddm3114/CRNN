import cv2
import csv 
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

# 构造 dataset
# pattern = 0(train) or 1(test) or 2(verify)
class MyDataset(torch.utils.data.Dataset):
	def __init__(self, pattern=0, train=0.8, test=0.05, # verify= 1 - train - test
			  csv_path=r'lpr.csv', dataset_path=r'cropped_lps/cropped_lps/'):
		super().__init__()
		assert pattern==0 or pattern==1 or pattern==2
		x = []
		y = []
		with open(csv_path, 'r') as file:  
			reader = csv.reader(file)  
			for row in reader:
				x.append(dataset_path + row[1])     # x记录了每个车牌图片的完整路径（可以直接打开）
				y.append(row[2])	                # y记录了每个车牌的车牌号（字符串)

		l = len(y)
		num = [0, int(l * train), int(l * (train+test)), l]
		self.x = x[num[pattern] : num[pattern+1]]
		self.y = y[num[pattern] : num[pattern+1]]

	def __getitem__(self, index):
		return fixed_size_tensor(self.x[index]) , self.y[index]
	
	def __len__(self):
		return len(self.y)

# 输入 path_name, 输出 tensor（大小为100，32）
# 将 dataset 中 x 的每个元素作为函数的输入值，返回其对应的tensor（已调整尺寸）
def fixed_size_tensor(path_name, x_size=32, y_size=100):
	"""
    image = Image.open(path_name).convert('L')
    original_image = torch.from_numpy(np.array(image))
    new_image = torch.zeros(x_size,y_size)
    image.close()

    original_x_size, original_y_size = original_image.shape[0:2]
    for i in range(x_size):
        for j in range(y_size):
            x = (original_x_size-1) * i / (x_size-1)
            y = (original_y_size-1) * j / (y_size-1)
            a = x-int(x)
            b = y-int(y)
            weight = torch.tensor([[(1-a)*(1-b),(1-a)*b],[a*(1-b),a*b]])
            new_image[i,j] = torch.multiply(weight, original_image[int(x):int(x)+2, int(y):int(y)+2]).sum() / 127.5 -1
    """
	image = cv2.imread(path_name)                            # 读取图片
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     	# 转化为灰度图
	image = cv2.bilateralFilter(image, 3, 0, 75)       		# 平滑化
	#image = cv2.Canny(image, 70, 400)                   	# 提取边界
	image = cv2.resize(image, (y_size,x_size), interpolation=cv2.INTER_LINEAR)
	transform = transforms.ToTensor()    
	tensor_image = transform(image)
	new_image = tensor_image / 127.5 - 1
	return  new_image.view(x_size,y_size)


if __name__ == '__main__':
	# test fixed_size_image func
	path = r'cropped_lps/cropped_lps/10002.jpg'
	image = Image.open(path).convert('L')
	plt.set_cmap('gray')
	plt.subplot(211)
	plt.imshow(image)
	image = Image.fromarray(np.array(fixed_size_tensor(path)))
	plt.subplot(212)
	plt.imshow(image)
	#plt.show()

	# test dataset
	train_data = 	MyDataset(pattern = 0)	# 0.80
	test_data =		MyDataset(pattern = 1)	# 0.05
	verify_data = 	MyDataset(pattern = 2)	# 0.15
	print(len(train_data), len(test_data), len(verify_data))
