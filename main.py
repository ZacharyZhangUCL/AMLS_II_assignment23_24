import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from A.main_A import DIV2KDataset_bicubic, CustomResNet, train_ResNet, test_ResNet
from B.main_B import DIV2KDataset_unknown, Generator, Discriminator, train_GAN, test_GAN


# ======================================================================================================================
# Hyperparameter settings
scale = 2  # Sample rate, {2, 3, 4} corespond to {X2, X3, X4}
batch_size = 2
learning_rate = 1e-4
num_epochs = 100
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])  # Normalization, "mean" and "std" are based on ImageNet
pretrain = True  # Whether to use pretrained model weights
cuda = 0
device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")  # "cuda" if torch.backends.mps.is_available() else "cpu"

# ======================================================================================================================
## Task A

# Data preprocessing.
file_root = "AMLSII_23-24_SN23058888/Datasets/"
train_dataset = DIV2KDataset_bicubic(file_root+'DIV2K_train_HR', file_root+'DIV2K_train_LR_bicubic', scale, transform=transform)
valid_dataset = DIV2KDataset_bicubic(file_root+'DIV2K_valid_HR', file_root+'DIV2K_valid_LR_bicubic', scale, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Build model object.
model = CustomResNet(scale).to(device)
model_root = f"AMLSII_23-24_SN23058888/A/model_bicubic_X{scale}.pth"
if pretrain == True:
    model.load_state_dict(torch.load(model_root))
else:
    # Model training.
    model, _, _ = train_ResNet(model, train_dataloader, valid_dataloader, num_epochs, 
                                            learning_rate, model_root, device=device)
# Train model based on the training set (you should fine-tune your model based on validation set.
_, _, acc_A_train = test_ResNet(model, train_dataloader)
# Test model based on the test set.
_, _, acc_A_test = test_ResNet(model, valid_dataloader)  

# Some code to free memory if necessary.
torch.cuda.empty_cache()             


# ======================================================================================================================
## Task B

# Data preprocessing.
file_root = "AMLSII_23-24_SN23058888/Datasets/"
train_dataset = DIV2KDataset_unknown(file_root+'DIV2K_train_HR', file_root+'DIV2K_train_LR_unknown', scale, transform=transform)
valid_dataset = DIV2KDataset_unknown(file_root+'DIV2K_valid_HR', file_root+'DIV2K_valid_LR_unknown', scale, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Build model object.
generator = Generator(scale).to(device)
discriminator = Discriminator().to(device)
model_gen_root = f"AMLSII_23-24_SN23058888/B/generator_unknown_X{scale}.pth"
model_disc_root = f"AMLSII_23-24_SN23058888/B/discriminator_unknown_X{scale}.pth"
if pretrain == True:
    generator.load_state_dict(torch.load(model_gen_root))
    # discriminator.load_state_dict(torch.load(model_disc_root))
else:
    # Model training.
    generator, _, train_losses_G, train_losses_D, val_losses_G = train_GAN(generator, discriminator, 
                                            train_dataloader, valid_dataloader, num_epochs, device, 
                                            model_gen_root, model_disc_root, learning_rate, patience=5)
# Train model based on the training set (you should fine-tune your model based on validation set.
_, _, acc_B_train = test_GAN(generator, train_dataloader)
# Test model based on the test set.
_, _, acc_B_test = test_GAN(generator, valid_dataloader)  
 
# Some code to free memory if necessary.
torch.cuda.empty_cache()     



# ======================================================================================================================
## Print out your results with following format:
print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test, acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'