import os
from PIL import Image
import torch
import cv2
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt

''' ---- This is a helper file. It contains functions that are used in main.py ----- '''


# This function finds the minimum width and height of all images in a directory.
def find_min_dimensions(train_dir, img1_extension):
    # Initialize minimum width and height to a large number
    min_width = min_height = float('inf')

    # Iterate over all images in the directory to find the smallest width and height
    for subdir, dirs, files in os.walk(train_dir):
        # print('subdir: ', subdir, '\ndirs: ', dirs, '\nfiles: ', files)
        for file in files:
            if file.endswith(img1_extension):  # Add or remove file extensions as needed
                img = Image.open(os.path.join(subdir, file))
                width, height = img.size
                min_width = min(min_width, width)
                min_height = min(min_height, height)

    print(f'Smallest width: {min_width}, Smallest height: {min_height}')
    return min_width, min_height


def prepare_data(min_height, min_width, downscale_factor, train_dir, eval_dir):
    # Define the transformations you want to apply to your images
    transform = transforms.Compose([
        transforms.Resize((min_height, min_width)),
        transforms.ToTensor()  # Convert image to PyTorch Tensor data type
    ])

    # Create an ImageFolder dataset
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    print('Dataset with label', train_dataset.classes, 'is ready')
    eval_dataset = ImageFolder(root=eval_dir, transform=transform)
    print('Dataset with label', eval_dataset.classes, 'is ready')

    # Define the transformations you want to apply to your images
    downscale_transform = transforms.Compose([
        # Downscale
        transforms.Resize((min_height // downscale_factor, min_width // downscale_factor)),
        # Upscale using bicubic interpolation
        transforms.Resize((min_height, min_width), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),  # Convert image to PyTorch Tensor data type
    ])

    # Create a new ImageFolder dataset for the downscaled and then bicubic interpolated images
    bicubic_dataset = ImageFolder(root=train_dir, transform=downscale_transform)
    print('Downscaled dataset with label', bicubic_dataset.classes, 'is ready')

    return train_dataset, eval_dataset, bicubic_dataset


def dataset_info(train_loader, eval_loader, train_dataset, eval_dataset):
    # Iterate over the data loader
    for i, (images, labels) in enumerate(train_loader):
        # `images` is a batch of images
        # `labels` is a batch of corresponding labels

        print(f'Batch {i + 1}:')
        print(f'Train dataset Image shape: {images.shape}')  # Should be [batch_size, num_channels, height, width]
        print(f'Train dataset Label shape: {labels.shape}')  # Should be [batch_size]

        # If you want to access the first image in the batch
        first_image = images[0]
        first_label = labels[0]

        # If you want to iterate over all images and labels in the batch
        for image, label in zip(images, labels):
            pass  # Replace with your code

    # Iterate over the data loader
    for i, (images, labels) in enumerate(eval_loader):
        # `images` is a batch of images
        # `labels` is a batch of corresponding labels

        print(f'Batch {i + 1}:')
        print(f'Evaluation dataset Image shape: {images.shape}')  # Should be [batch_size, num_channels, height, width]
        print(f'Evaluation dataset Label shape: {labels.shape}')  # Should be [batch_size]

    print(f'The training dataset has {len(train_dataset)} images.')
    print(f'The evaluation dataset has {len(eval_dataset)} images.')


def print_data(train_dataset, bicubic_dataset, eval_dataset, train_loader, eval_loader, batch_size):
    image, label = train_dataset[0]
    print('--------------------------------------------------------------------------------------------------')
    print('-----------------------------------TRAINING DATASET INFORMATION-----------------------------------')
    print('--------------------------------------------------------------------------------------------------')

    print('Image shape: ', image.shape)
    print('Image type: ', image.dtype, '\nLabel type: ', type(label))
    print('Image min/max value: ', image.min().item(), '/', image.max().item())
    print('\nLabel=', label)

    print('--------------------------------------------------------------------------------------------------')
    print('-----------------------------------EVALUATION DATASET INFORMATION---------------------------------')
    print('--------------------------------------------------------------------------------------------------')

    image, label = eval_dataset[0]
    print('Image shape: ', image.shape)
    print('Image type: ', image.dtype, '\nLabel type: ', type(label))
    print('Image min/max value: ', image.min().item(), '/', image.max().item())
    print('\nLabel=', label)

    print('--------------------------------------------------------------------------------------------------')
    print('-----------------------------------BICUBIC DATASET INFORMATION------------------------------------')
    print('--------------------------------------------------------------------------------------------------')

    image, label = bicubic_dataset[0]
    print('Image shape: ', image.shape)
    print('Image type: ', image.dtype, '\nLabel type: ', type(label))
    print('Image min/max value: ', image.min().item(), '/', image.max().item())
    print('\nLabel=', label)

    print('--------------------------------------------------------------------------------------------------')
    print('-----------------------------------TRAINING LOADER INFORMATION------------------------------------')
    print('--------------------------------------------------------------------------------------------------')
    print('Train loader length: ', len(train_loader))
    print('Train loader batch size: ', train_loader.batch_size)
    print('Train loader dataset length: ', len(train_loader))
    print('Train loader number of batches: ', len(train_loader) / train_loader.batch_size)
    print('Train loader number of batches rounded: ', round(len(train_loader) / train_loader.batch_size))
    print('Train loader number of batches rounded up: ', round(len(train_loader) / train_loader.batch_size) + 1)
    print('Train loader number of batches rounded down: ', round(len(train_loader) / train_loader.batch_size) - 1)

    #print('--------------------------------------------------------------------------------------------------')
    #print('-----------------------------------BICUBIC LOADER INFORMATION------------------------------------')
    #print('--------------------------------------------------------------------------------------------------')
    #print('Bicubic loader length: ', len(bicubic_loader))
    #print('Bicubic loader batch size: ', bicubic_loader.batch_size)
    #print('Bicubic loader dataset length: ', len(bicubic_loader))
    #print('Bicubic loader number of batches: ', len(bicubic_loader) / bicubic_loader.batch_size)
    #print('Bicubic loader number of batches rounded: ', round(len(bicubic_loader) / bicubic_loader.batch_size))
    #print('Bicubic loader number of batches rounded up: ', round(len(bicubic_loader) / bicubic_loader.batch_size) + 1)
    #print('Bicubic loader number of batches rounded down: ', round(len(bicubic_loader) / bicubic_loader.batch_size) - 1)

    print('--------------------------------------------------------------------------------------------------')
    print('-----------------------------------EVALUATION LOADER INFORMATION----------------------------------')
    print('--------------------------------------------------------------------------------------------------')
    print('Eval loader length: ', len(eval_loader))
    print('Eval loader batch size: ', eval_loader.batch_size)
    print('Eval loader dataset length: ', len(eval_loader))
    print('Eval loader number of batches: ', len(eval_loader) / eval_loader.batch_size)
    print('Eval loader number of batches rounded: ', round(len(eval_loader) / eval_loader.batch_size))
    print('Eval loader number of batches rounded up: ', round(len(eval_loader) / eval_loader.batch_size) + 1)
    print('Eval loader number of batches rounded down: ', round(len(eval_loader) / eval_loader.batch_size) - 1)


def calculate_ssim(img1, img2, height, width):
    # Ensure the images are floating point tensors
    img1 = img1.float()
    img2 = img2.float()

    # Reshape the 1D tensors to 2D
    img1 = img1.view(height, width)
    img2 = img2.view(height, width)

    # Convert PyTorch tensors to NumPy arrays
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()

    # Calculate SSIM
    ssim_value = compare_ssim(img1_np, img2_np, data_range=img1_np.max() - img1_np.min())

    return ssim_value

def rgb_to_ycbcr(image):
    # Convert the tensor to a PIL Image
    to_pil_image = ToPILImage()
    pil_image = to_pil_image(image)

    # Convert the PIL Image to YCbCr
    ycbcr_image = pil_image.convert('YCbCr')
    y, cb, cr = ycbcr_image.split()
    return y, cb, cr


def _check_images(img1: np.ndarray, img2: np.ndarray):
   assert img1.shape == img2.shape, "Input images must have the same dimensions."
   assert img1.dtype == img2.dtype, "Input images must have the same data type."


def ssim(raw_image: np.ndarray, dst_image: np.ndarray, crop_border: int) -> float:
    """Python implements the SSIM (Structural Similarity) function, which calculates single-channel data

    Args:
        raw_image (np.ndarray): The image data to be compared, in Y channel format, the data range is [0, 255]
        dst_image (np.ndarray): reference image data, Y channel format, data range is [0, 255]
        crop_border (int): crop border a few pixels

    Returns:
        ssim_metrics (float): SSIM metrics for single channel

    """
    # crop border pixels
    if crop_border > 0:
        raw_image = raw_image[crop_border:-crop_border, crop_border:-crop_border]
        dst_image = dst_image[crop_border:-crop_border]

    # Convert data type to numpy.float64 bit
    raw_image = raw_image.astype(np.float64)
    dst_image = dst_image.astype(np.float64)

    # Calculate SSIM for the Y channel
    ssim_metrics = _ssim(raw_image, dst_image)

    return ssim_metrics


def _ssim(raw_image: np.ndarray, dst_image: np.ndarray) -> float:
    """Python implements the SSIM (Structural Similarity) function, which only calculates single-channel data

    Args:
        raw_image (np.ndarray): The image data to be compared, in Y channel format, the data range is [0, 255]
        dst_image (np.ndarray): reference image data, Y channel format, data range is [0, 255]

    Returns:
        ssim_metrics (float): SSIM metrics for single channel

    """
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    kernel = cv2.getGaussianKernel(11, 1.5)
    kernel_window = np.outer(kernel, kernel.transpose())

    raw_mean = cv2.filter2D(raw_image, -1, kernel_window)[5:-5]
    dst_mean = cv2.filter2D(dst_image, -1, kernel_window)[5:-5]

    raw_mean_square = raw_mean ** 2
    dst_mean_square = dst_mean ** 2
    raw_dst_mean = raw_mean * dst_mean

    raw_variance = cv2.filter2D(raw_image ** 2, -1, kernel_window)[5:-5] - raw_mean_square
    dst_variance = cv2.filter2D(dst_image ** 2, -1, kernel_window)[5:-5] - dst_mean_square
    raw_dst_covariance = cv2.filter2D(raw_image * dst_image, -1, kernel_window)[5:-5] - raw_dst_mean

    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (raw_variance + dst_variance + c2)

    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = np.mean(ssim_metrics)

    return ssim_metrics


def ssim_torch(y_true, y_pred):
    C1 = 0.01**2
    C2 = 0.03**2

    mu_y_true = torch.mean(y_true)
    mu_y_pred = torch.mean(y_pred)
    std_y_true = torch.std(y_true)
    std_y_pred = torch.std(y_pred)
    cov = torch.mean((y_true - mu_y_true) * (y_pred - mu_y_pred))

    numerator = (2 * mu_y_true * mu_y_pred + C1) * (2 * cov + C2)
    denominator = (mu_y_true**2 + mu_y_pred**2 + C1) * (std_y_true**2 + std_y_pred**2 + C2)

    return numerator / denominator


# Create a new dataset with only the Y channel images
class YChannelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        y, _, _ = rgb_to_ycbcr(image)
        return ToTensor()(y)


# Create a new dataset with only the Cb or Cr channel images
class CbCrChannelDataset(Dataset):
    def __init__(self, dataset, channel):
        self.dataset = dataset
        self.channel = channel  # 1 for Cb, 2 for Cr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        _, cb, cr = rgb_to_ycbcr(image)
        channel_image = cb if self.channel == 1 else cr
        return ToTensor()(channel_image)

class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        assert len(dataset1) == len(dataset2)
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        return x1, x2

    def __len__(self):
        return len(self.dataset1)
