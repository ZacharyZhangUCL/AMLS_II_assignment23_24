import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np


# Dataset
class DIV2KDataset_bicubic(Dataset):
    def __init__(self, hr_root_dir, lr_root_dir, scale:int, transform=None):
        """
        初始化DIV2K数据集。
        hr_root_dir: 高分辨率图像的根目录。
        lr_root_dir: 低分辨率图像的根目录。
        scale: 下采样的比例(X2, X3, X4)。
        transform: 应用于图像的转换。
        """
        self.hr_root_dir = hr_root_dir
        self.lr_root_dir = os.path.join(lr_root_dir, f'X{scale}')
        self.transform = transform
        self.hr_images = os.listdir(hr_root_dir)
        self.lr_images = os.listdir(self.lr_root_dir)
        self.scale = scale

        # 对文件名进行排序，以确保匹配
        self.hr_images = sorted(os.listdir(hr_root_dir))
        self.lr_images = sorted(os.listdir(self.lr_root_dir))

    def __len__(self):
        """
        返回数据集中图像的数量。
        """
        return len(self.hr_images)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的图像。
        """
        hr_image_path = os.path.join(self.hr_root_dir, self.hr_images[idx])
        lr_image_path = os.path.join(self.lr_root_dir, self.lr_images[idx])

        # # 打印文件名以检查是否匹配
        # print(f"HR image: {self.hr_images[idx]}, LR image: {self.lr_images[idx]}")

        hr_image = Image.open(hr_image_path).resize((2040, 2040))
        lr_image = Image.open(lr_image_path).resize((2040 // self.scale, 2040 // self.scale))

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        """
        in_channels: 输入的通道数。
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


# 自定义ResNet
class CustomResNet(nn.Module):
    def __init__(self, scale:int):
        """
        scale: 下采样的比例(X2, X3, X4)。
        """
        super(CustomResNet, self).__init__()
        in_channels = 64
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True) # 将图像从(2040 // scale) x (2040 // scale)上采样到2040x2040
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=1, padding=3) # 初始卷积层: 输出格式 2040x2040x64
        self.relu = nn.ReLU(inplace=True)
        self.resblock1 = ResidualBlock(in_channels) # 残差块1: 输出格式 2040x2040x64
        self.resblock2 = ResidualBlock(in_channels) # 残差块2: 输出格式 2040x2040x64
        self.resblock3 = ResidualBlock(in_channels) # 残差块3: 输出格式 2040x2040x64
        self.resblock4 = ResidualBlock(in_channels) # 残差块4: 输出格式 2040x2040x64
        self.conv2 = nn.Conv2d(in_channels, 3, kernel_size=7, stride=1, padding=3) # 最后的卷积层，恢复到3个通道: 输出格式 2040x2040x3

    def forward(self, x):
        x = self.upsample(x) # 上采样到2040x2040
        x = self.conv1(x)
        x = self.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.conv2(x)
        return x


# 模型训练
def train_ResNet(model, train_loader, val_loader, num_epochs, learning_rate, model_root, patience=5,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
                # "cuda" if torch.backends.mps.is_available() else "cpu"
    """
    训练模型的函数。
    model: 要训练的模型。
    train_loader: 训练数据加载器。
    val_loader: 验证数据加载器。
    num_epochs: 训练的轮数。
    learning_rate: 学习率。
    model_root: 模型保存路径。
    patience: 早停耐心参数。
    device: 运行的设备。
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # 退火学习率，每个epoch学习率衰减5%

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 用于存储训练和验证损失
    train_losses, val_losses = [], []

    print("Training Starts!")

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0

        # 遍历训练数据
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 清除之前的梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证过程
        model.eval()  # 设置模型为评估模式
        running_loss = 0.0

        with torch.no_grad():  # 不计算梯度，节省内存和计算资源
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

        # 计算平均验证损失
        avg_val_loss = running_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 打印本轮的训练信息
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4e}, Val Loss: {avg_val_loss:.4e}")

        # 退火学习率调整
        scheduler.step()

        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存最佳生成器模型
            torch.save(model.state_dict(), model_root)
            best_model = model
            best_train_loss, best_val_losses = avg_train_loss, avg_val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch + 1} epochs! Train Loss: {best_train_loss:.4e}, Val Loss: {best_val_losses:.4e}')
                break
    
    print("Training Ends!")
    model = best_model

    return model, train_losses, val_losses


# 反归一化操作
def unnormalize(tensor, mean, std):
    if tensor.is_cuda:
        tensor = tensor.cpu()  # 确保张量在CPU上
    tensor = tensor.numpy().transpose((1, 2, 0))
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor


# 计算PSNR和SSIM
def calculate_psnr_ssim(hr_image, generated_image, mean, std):
    hr_image = unnormalize(hr_image, mean, std)
    generated_image = unnormalize(generated_image, mean, std)
    psnr = compare_psnr(hr_image, generated_image, data_range=255)
    ssim = compare_ssim(hr_image, generated_image, channel_axis=-1)
    return psnr, ssim


# 模型测试函数
def test_ResNet(model, test_loader, mean, std, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    criterion = nn.MSELoss()
    total_psnr, total_ssim, running_loss = 0.0, 0.0, 0.0
    num_samples = 0

    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
            generated_hr = model(lr_imgs)
            loss = criterion(generated_hr, hr_imgs)
            running_loss += loss.item()

            # 计算PSNR和SSIM
            for i in range(hr_imgs.size(0)):
                psnr, ssim = calculate_psnr_ssim(hr_imgs[i], generated_hr[i], mean, std)
                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1

    test_loss = running_loss / len(test_loader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print(f"Test Loss: {test_loss:.4e}, Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")

    return test_loss, avg_psnr, avg_ssim


# 绘制训练过程图
def image_training_process(train_losses, val_losses, scale):
    filename=f'AMLSII_23-24_SN23058888/A/Task_A_Training_Process_bicubic_X{scale}.png'
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


# 绘制数据集示例图
def plot_dataset_samples(loader, scale, mean, std, dpi=2000):
    """
    绘制并保存数据集示例图。
    loader: 数据加载器。
    filename: 图片保存的文件名。
    """
    filename=f'AMLSII_23-24_SN23058888/A/Task_A_Dataset_Samples_bicubic_X{scale}.png'
    lr, hr = next(iter(loader))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(unnormalize(lr[0], mean, std))
    axs[0].set_title('Low Resolution')
    axs[0].axis('off')

    axs[1].imshow(unnormalize(hr[0], mean, std))
    axs[1].set_title('High Resolution')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)


# 绘制实验结果图
def plot_validation_results(model, val_loader, scale, mean, std, dpi=2000):
    """
    从验证集中获取一张图像，通过模型生成结果，并展示原始的低分辨率和高分辨率图像与模型生成的图像。
    model: 训练好的模型。
    val_loader: 验证集的数据加载器。
    filename: 结果保存的文件名。
    """
    
    filename=f'AMLSII_23-24_SN23058888/A/Task_A_Results_Comparison_bicubic_X{scale}.png'

    # 获取验证集中的一对图像
    lr, hr = next(iter(val_loader))

    # 检查模型所在设备，并将数据移至相同设备
    device = next(model.parameters()).device
    lr, hr = lr.to(device), hr.to(device)

    with torch.no_grad():
        model.eval()
        sr = model(lr[0].unsqueeze(0)).squeeze(0)

    # 反归一化
    lr_img = unnormalize(lr[0], mean, std)
    hr_img = unnormalize(hr[0], mean, std)
    sr_img = unnormalize(sr, mean, std)

    # 绘制图像
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title('Low Resolution')
    axs[0].axis('off')

    axs[1].imshow(hr_img)
    axs[1].set_title('High Resolution (Original)')
    axs[1].axis('off')

    axs[2].imshow(sr_img)
    axs[2].set_title('Super-Resolved')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)


def main_A(scale:int):
    # 超参数设置
    cuda = 3
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 100
    pretrain = True  # 是否使用预训练模型权重
    mean=[0.485, 0.456, 0.406]# 数据转换，这里的均值（mean）和标准差（std）是基于ImageNet数据集的图像计算得出的
    std=[0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")  # "cuda" if torch.backends.mps.is_available() else "cpu"

    # 创建数据集
    file_root = "AMLSII_23-24_SN23058888/Datasets/"
    train_dataset = DIV2KDataset_bicubic(file_root+'DIV2K_train_HR', file_root+'DIV2K_train_LR_bicubic', scale, transform=transform)
    valid_dataset = DIV2KDataset_bicubic(file_root+'DIV2K_valid_HR', file_root+'DIV2K_valid_LR_bicubic', scale, transform=transform)

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 构建模型
    model = CustomResNet(scale).to(device)
    model_root = f"AMLSII_23-24_SN23058888/A/model_bicubic_X{scale}.pth"
    if pretrain == True:
        model.load_state_dict(torch.load(model_root))

    # 训练模型
    # model, train_losses, val_losses = train_ResNet(model, train_dataloader, valid_dataloader, num_epochs, 
    #                                        learning_rate, model_root, device=device)

    # # 测试模型
    # test_loss, avg_psnr, avg_ssim = test_ResNet(model, valid_dataloader, mean, std, device)


    # # 绘制训练过程图
    # image_training_process(train_losses, val_losses, scale)

    # 绘制数据集示例图
    plot_dataset_samples(train_dataloader, scale, mean, std)

    # 绘制实验结果图
    plot_validation_results(model, valid_dataloader, scale, mean, std)

if __name__== "__main__" :
    for scale in {2,3,4}:
        main_A(scale)