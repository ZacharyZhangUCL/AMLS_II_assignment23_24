# README
# Super-Resolution Deep Learning Models

This README provides an overview of the project structure, detailing the roles of each file and the packages required to run the code.

## Project Organization

The project is organized into several folders, each dedicated to a specific subtask of the super-resolution challenge.

- `Datasets`: This folder is intended for storing DIV2K Datasets, including `DIV2K_train_HR`, `DIV2K_train_LR_bicubic`, `DIV2K_train_LR_unknown`, `DIV2K_valid_HR`, `DIV2K_valid_LR_bicubic`, `DIV2K_valid_LR_unknown`
- `A`, `B`: Each folder corresponds to a specific subtask, including standard super-resolution (`Task A`) and unknown down-sampling super-resolution (`Task B`). The folders contain the respective code files.

## Role of Each File

- `main.py`: The main function file orchestrates the overall process and integrates various components of the project.
- `main_A.py`: The primary function file for Task A (standard super-resolution), which includes the implementation of the ResNet model and its training and evaluation process.
- `main_B.py`: The main function file for Task B (unknown down-sampling super-resolution) that comprises the GAN model's setup, training, and testing routines.

## Packages Required

To run the code, the following packages are required:
- `os`: For directory and file operations.
- `PIL.Image`: For image processing tasks.
- `torch`: The main PyTorch package for deep learning model implementation.
- `torch.utils.data`: For dataset creation and data loading utilities.
- `torchvision`: For transformations and dataset handling specific to images.
- `torch.optim`: For optimization algorithms like Adam.
- `torch.nn`: For neural network layers and functions.
- `matplotlib.pyplot`: For plotting graphs and images.
- `numpy`: For numerical operations.
- `skimage.metrics`: For calculating metrics like peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM).

To install these packages, you can use the following command:
```sh
pip install pillow torch torchvision matplotlib numpy scikit-image
```

Make sure to activate your virtual environment before installing if you are using one.

## Execution

To run the main function, you can use the following command:
```sh
python main.py
```
