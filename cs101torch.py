import os
import cv2
import imageio.v2 as imageio 
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# 加载数据
A_data = loadmat("E:/ucd life/lab/xue lab/project/cs/A.mat")
Y_data = loadmat("E:/ucd life/lab/xue lab/project/cs/Y.mat")

A = A_data['A']
Y = Y_data['Y']

# 转换为PyTorch张量并创建数据集
A_tensor = torch.tensor(A, dtype=torch.float32)/255
Y_tensor = torch.tensor(Y, dtype=torch.float32)
dataset = TensorDataset(A_tensor, Y_tensor)

# 划分训练集和验证集（80%训练，20%验证）
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 180
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义模型（无隐藏层）
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        #self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.fc1(x)
        #x = self.sigmoid(x) 
        return x

# 调整模型参数
input_size = 10000
output_size = 1
model = SimpleNN(input_size, output_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数：MSE + TV Loss
def total_variation_loss(weight_matrix):
    """
    计算权重矩阵的 Total Variation Loss
    :param weight_matrix: PyTorch Tensor, 权重矩阵 [H, W]
    :return: TV Loss
    """
    dx = torch.abs(weight_matrix[:, :-1] - weight_matrix[:, 1:]).sum()
    dy = torch.abs(weight_matrix[:-1, :] - weight_matrix[1:, :]).sum()
    return dx + dy

criterion = nn.MSELoss()
tv_lambda = 5e-4  # TV 正则化系数
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# 创建保存图像的文件夹
os.makedirs("epoch_images", exist_ok=True)

# 训练循环
epochs = 10000
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_A, batch_Y in train_loader:
        batch_A, batch_Y = batch_A.to(device), batch_Y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_A)
        mse_loss = criterion(outputs, batch_Y)

        #TVloss
        W1 = model.fc1.weight
        W1 = W1.reshape(100, 100)
        tv_loss = tv_lambda * total_variation_loss(W1)

        loss = mse_loss + tv_loss

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_A.size(0)
    
    # 计算验证损失
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_A, batch_Y in val_loader:
            batch_A, batch_Y = batch_A.to(device), batch_Y.to(device)
            outputs = model(batch_A)
            mse_loss = criterion(outputs, batch_Y)

            W1 = model.fc1.weight
            W1 = W1.reshape(100, 100)
            tv_loss = tv_lambda * total_variation_loss(W1)

            loss = mse_loss + tv_loss

            val_loss += loss.item() * batch_A.size(0)
    
    train_loss = train_loss / len(train_dataset)
    val_loss = val_loss / len(val_dataset)
    scheduler.step(val_loss)  # 调整学习率
    
    # 打印 Loss（科学计数法）
    
    
    # 每 10 个 Epoch 生成一张图
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, TV Loss{tv_loss:.4e}")
        W1 = model.fc1.weight.detach().cpu().numpy()
        X = W1.reshape(100, 100)
        X_normalized = (X - np.min(X)) / (np.max(X) - np.min(X))
        
        plt.imshow(X_normalized, cmap='gray', vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"Epoch {epoch+1} Weights (100x100)")
        plt.savefig(f"epoch_images/epoch_{epoch+1}.png")
        plt.close()

frame_size = (800, 600)  # 视频帧大小
fps = 2  # 每秒帧数
video_name = "weights_evolution.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

for epoch in range(10, epochs + 1, 10):
    img_path = f"epoch_images/epoch_{epoch}.png"
    img = cv2.imread(img_path)
    if img is not None:
        resized_img = cv2.resize(img, frame_size)
        video_writer.write(resized_img)

video_writer.release()
print(f"Video saved as {video_name}")
