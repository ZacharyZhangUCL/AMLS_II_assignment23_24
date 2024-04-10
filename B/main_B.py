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
class DIV2KDataset_unknown(Dataset):
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


# 生成器
class Generator(nn.Module):
    def __init__(self, scale:int):
        """
        生成器网络，用于上采样低分辨率图像到高分辨率。
        scale_factor: 上采样因子。
        """
        super(Generator, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        in_channels = 64
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=1, padding=3)
        self.relu = nn.ReLU(inplace=True)
        # 一系列残差块，以提升网络性能
        self.resblock1 = ResidualBlock(in_channels)
        self.resblock2 = ResidualBlock(in_channels)
        self.resblock3 = ResidualBlock(in_channels)
        self.resblock4 = ResidualBlock(in_channels)
        self.conv2 = nn.Conv2d(in_channels, 3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x = self.upsample(x)  # 上采样至高分辨率
        x = self.conv1(x)
        x = self.relu(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.conv2(x)
        return x


# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # 第二层卷积层
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.leakyrelu2 = nn.LeakyReLU(0.2, inplace=True)

        # 第三层卷积层
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.leakyrelu3 = nn.LeakyReLU(0.2, inplace=True)

        # 第四层卷积层
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.leakyrelu4 = nn.LeakyReLU(0.2, inplace=True)

        # 最后一个卷积层，用于输出判别结果
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leakyrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leakyrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leakyrelu4(x)

        x = self.conv5(x)
        x = x.mean(dim=(2, 3))  # 全局平均池化，将输出维度从 [B, C, H, W] 缩减为 [B, C]

        return x


# 模型训练
def train_GAN(generator, discriminator, train_loader, val_loader, num_epochs, device,
              model_gen_root, model_disc_root, learning_rate=1e-4, patience=5):

    # 定义生成器和判别器的优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate*1e-1)

    # 初始化余弦退火调度器
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=learning_rate*1e-3)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=learning_rate*1e-4)

    # 定义对抗损失和内容损失
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_content = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # 用于存储训练和验证损失
    train_losses_G, train_losses_D, val_losses_G = [], [], []

    print("Training Starts!")

    for epoch in range(num_epochs):
        train_loss_G, train_loss_D = 0.0, 0.0

        for lr_imgs, hr_imgs in train_loader:
            # 数据搬移到设备上
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            valid = torch.ones((hr_imgs.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((hr_imgs.size(0), 1), requires_grad=False).to(device)

            # 训练生成器
            optimizer_G.zero_grad()

            generated_hr = generator(lr_imgs)
            pred_fake = discriminator(generated_hr)
            loss_GAN = criterion_GAN(pred_fake, valid)
            loss_content = criterion_content(generated_hr, hr_imgs)

            loss_G = loss_GAN + loss_content
            loss_G.backward()
            optimizer_G.step()

            train_loss_G += loss_G.item()

            # 训练判别器
            optimizer_D.zero_grad()

            pred_real = discriminator(hr_imgs)
            loss_real = criterion_GAN(pred_real, valid)

            pred_fake = discriminator(generated_hr.detach())
            loss_fake = criterion_GAN(pred_fake, fake)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            train_loss_D += loss_D.item()

        # 计算本轮的平均损失
        avg_train_loss_G = train_loss_G / len(train_loader)
        avg_train_loss_D = train_loss_D / len(train_loader)
        train_losses_G.append(avg_train_loss_G)
        train_losses_D.append(avg_train_loss_D)

        # 验证生成器的性能
        generator.eval()
        val_loss_G = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)

                generated_hr = generator(lr_imgs)
                val_loss_G += criterion_content(generated_hr, hr_imgs).item()

        avg_val_loss = val_loss_G / len(val_loader)
        val_losses_G.append(avg_val_loss)
        
        # 打印本轮的训练信息
        print(f'Epoch [{epoch+1}/{num_epochs}], Generator Loss: {avg_train_loss_G:.4e}, Discriminator Loss: {avg_train_loss_D:.4e}, Val Loss: {avg_val_loss:.4e}')

        # 退火学习率调整
        scheduler_G.step()
        scheduler_D.step()

        # 检查是否需要早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # 保存最佳生成器模型和判别器模型
            torch.save(generator.state_dict(), model_gen_root)
            torch.save(discriminator.state_dict(), model_disc_root)
            best_generator = generator
            best_discriminator = discriminator
            best_train_loss_G, best_train_loss_D, best_val_loss = avg_train_loss_G, avg_train_loss_D, avg_val_loss
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f'Early stopping triggered after {epoch + 1} epochs! Generator Loss: {best_train_loss_G:.4e}, Discriminator Loss: {best_train_loss_D:.4e}, Val Loss: {best_val_loss:.4e}')
                break

    print("Training Ends!")
    generator = best_generator
    discriminator = best_discriminator

    return generator, discriminator, train_losses_G, train_losses_D, val_losses_G


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
def test_GAN(model, test_loader, mean, std, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
def image_training_process(train_losses_G, train_losses_D, val_losses_G, scale):
    filename = f'AMLSII_23-24_SN23058888/B/Task_B_Training_Process_unknown_X{scale}.png'
    
    plt.figure(figsize=(10, 6))
    
    # 绘制生成器的训练和验证损失
    plt.plot(train_losses_G, label='Generator Training Loss', color='blue')
    plt.plot(val_losses_G, label='Generator Validation Loss', color='cyan')
    
    # 绘制判别器的训练损失
    plt.plot(train_losses_D, label='Discriminator Training Loss', color='red')
    
    plt.title('Training and Validation Loss for Generator and Discriminator')
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
    filename=f'AMLSII_23-24_SN23058888/B/Task_B_Dataset_Samples_unknown_X{scale}.png'
    lr, hr = next(iter(loader))
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(unnormalize(lr[0], mean, std))
    axs[0].set_title('Low Resolution')
    axs[0].axis('off')

    axs[1].imshow(unnormalize(hr[0], mean, std))
    axs[1].set_title('High Resolution')
    axs[1].axis('off')

    plt.savefig(filename, dpi=dpi)


def plot_validation_results(generator, val_loader, scale, mean, std, dpi=2000):
    """
    从验证集中获取一张图像，通过生成器模型生成结果，并展示原始的低分辨率和高分辨率图像与生成器生成的图像。
    generator: 训练好的生成器模型。
    val_loader: 验证集的数据加载器。
    scale: 上采样的比例。
    mean, std: 用于反归一化的均值和标准差。
    dpi: 图像保存的分辨率。
    """
    
    filename = f'AMLSII_23-24_SN23058888/B/Task_B_Results_Comparison_unknown_X{scale}.png'

    lr, hr = next(iter(val_loader))  # 获取一对低分辨率和高分辨率图像

    device = next(generator.parameters()).device
    lr = lr.to(device)

    with torch.no_grad():
        generator.eval()
        sr = generator(lr[0].unsqueeze(0)).squeeze(0)  # 生成超分辨率图像

    # 反归一化
    lr_img = unnormalize(lr[0], mean, std)
    hr_img = unnormalize(hr[0], mean, std)
    sr_img = unnormalize(sr, mean, std)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(lr_img)
    axs[0].set_title('Low Resolution')
    axs[0].axis('off')

    axs[1].imshow(hr_img)
    axs[1].set_title('High Resolution (Original)')
    axs[1].axis('off')

    axs[2].imshow(sr_img)
    axs[2].set_title('Super-Resolved by GAN')
    axs[2].axis('off')

    plt.savefig(filename, dpi=dpi)


def main_B(scale:int):
    # 超参数设置
    cuda = 3
    batch_size = 2
    learning_rate = 1e-5
    num_epochs = 100
    pretrain = False  # 是否使用预训练模型权重
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")

    # 创建数据集和数据加载器
    file_root = "AMLSII_23-24_SN23058888/Datasets/"
    train_dataset = DIV2KDataset_unknown(file_root+'DIV2K_train_HR', file_root+'DIV2K_train_LR_unknown', scale, transform=transform)
    valid_dataset = DIV2KDataset_unknown(file_root+'DIV2K_valid_HR', file_root+'DIV2K_valid_LR_unknown', scale, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 构建生成器和判别器
    generator = Generator(scale).to(device)
    discriminator = Discriminator().to(device)
    
    # 模型保存路径
    model_gen_root = f"AMLSII_23-24_SN23058888/B/generator_unknown_X{scale}.pth"
    model_disc_root = f"AMLSII_23-24_SN23058888/B/discriminator_unknown_X{scale}.pth"

    # 如果使用预训练模型
    if pretrain == True:
        generator.load_state_dict(torch.load(model_gen_root))
        discriminator.load_state_dict(torch.load(model_disc_root))

    # 训练GAN模型
    generator, _, train_losses_G, train_losses_D, val_losses_G = train_GAN(generator, discriminator, 
                                        train_dataloader, valid_dataloader, num_epochs, device, 
                                        model_gen_root, model_disc_root, learning_rate, patience=5)

    # 测试模型
    test_loss, avg_psnr, avg_ssim = test_GAN(generator, valid_dataloader, mean, std, device)
    
    # 绘制训练过程图
    image_training_process(train_losses_G, train_losses_D, val_losses_G, scale)

    # 绘制数据集示例图
    plot_dataset_samples(valid_dataloader, scale, mean, std)

    # 绘制实验结果图
    plot_validation_results(generator, valid_dataloader, scale, mean, std)

if __name__ == "__main__":
    for scale in {2,3,4}:
        main_B(scale)
