import os
import torch
from torch import nn, optim
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib.pyplot as plt
from brisque import BRISQUE
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from matplotlib import patches
from matplotlib.transforms import Bbox

# ------------------------- 0.SETUP ------------------------------------------------------------------ #
# Define the directories for the training and evaluation datasets and get folder item extensions
train_dir = 'your_path_to_DIV2K_train_HR'
eval_dir = 'your_path_to_DIV2K_valid_HR'
first_image = 'your_path_to_DIV2K_train_HR_0001.png'
img1_name, img1_extension = os.path.splitext(first_image)
# Initialize the BRISQUE model
brisque = BRISQUE()
# ---------------- 1.FIND MINIMUM WIDTH AND HEIGHT -------------------------------------------------- #
# The purpose is to ensure stability of the model by ensuring that all images are of the same size

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

# ---------------- 2.CREATE DATASETS -------------------------------------------------------------- #

downscale_factor = 4
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

# Create a new ImageFolder dataset for the downscaled and upscaled images
test_dataset = ImageFolder(root=train_dir, transform=downscale_transform)
print('Downscaled dataset with label', test_dataset.classes, 'is ready')

# Get the first image from each dataset
original_image, original_label = train_dataset[6]
transformed_image, transformed_label = test_dataset[6]

print('og image size: ', original_image.size())
print('tr image size: ', transformed_image.size())

# Convert the tensors to numpy arrays for visualization
original_image_np = original_image.permute(1, 2, 0).numpy()
transformed_image_np = transformed_image.permute(1, 2, 0).numpy()

# Create a new figure
plt.figure(figsize=(20, 10))

# Display the original image
plt.subplot(1, 2, 1)
plt.title('Ground Truth')
plt.imshow(original_image_np)

# Display the transformed image
plt.subplot(1, 2, 2)
plt.title('Bicubic Interpolation')
plt.imshow(transformed_image_np)

# Show the figure
plt.show()

# Convert the tensors to PIL Images
original_image_pil = transforms.ToPILImage()(original_image)
transformed_image_pil = transforms.ToPILImage()(transformed_image)

# Convert the PIL Images to YCbCr color space
original_image_ycbcr = original_image_pil.convert('YCbCr')
transformed_image_ycbcr = transformed_image_pil.convert('YCbCr')

# Split the images into Y, Cb, and Cr channels
original_channels = original_image_ycbcr.split()
transformed_channels = transformed_image_ycbcr.split()

# Create a new figure
plt.figure(figsize=(10, 15))

# Display the Y, Cb, Cr channels of the original image
for i, channel in enumerate(original_channels):
    plt.subplot(3, 2, i*2 + 1)
    plt.title('Original Image - ' + ['Y', 'Cb', 'Cr'][i] + ' Channel')
    plt.imshow(channel, cmap='gray')

# Display the Y, Cb, Cr channels of the transformed image
for i, channel in enumerate(transformed_channels):
    plt.subplot(3, 2, i*2 + 2)
    plt.title('Transformed Image - ' + ['Y', 'Cb', 'Cr'][i] + ' Channel')
    plt.imshow(channel, cmap='gray')

# Show the figure
plt.show()

# Define the SRCNN model
class SRCNNu(nn.Module):
    def __init__(self):
        super(SRCNNu, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# Initialize the model
model = SRCNNu()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Get the Y channel of the first image from each dataset
original_image_y = original_channels[0]
transformed_image_y = transformed_channels[0]

# Convert the PIL Images to PyTorch tensors and add an extra batch dimension
original_image_y = transforms.ToTensor()(original_image_y).unsqueeze(0)
transformed_image_y = transforms.ToTensor()(transformed_image_y).unsqueeze(0)

# Move the images to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
original_image_y = original_image_y.to(device)
transformed_image_y = transformed_image_y.to(device)
# Train the model
num_epochs = 500
model.train()
for epoch in range(num_epochs):  # Change this to your desired number of epochs
    # Forward pass
    outputs = model(transformed_image_y)
    loss = criterion(outputs, original_image_y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/10000], Loss: {loss.item()}')

# Use the trained model to predict the Y channel of the transformed image
model.eval()
with torch.no_grad():
    predicted_y = model(transformed_image_y)

# Convert the predicted Y channel back to a PIL Image
predicted_y = transforms.ToPILImage()(predicted_y.squeeze(0))

# Get the Cb and Cr channels of the transformed image
_, cb, cr = transformed_channels

# Merge the predicted Y channel with the original Cb and Cr channels
reconstructed_image = Image.merge('YCbCr', [predicted_y, cb, cr])

# Convert the image back to RGB color space
reconstructed_image = reconstructed_image.convert('RGB')

# Save the reconstructed image
reconstructed_image.save('try_image.png')

# Display the original, transformed, and reconstructed images side by side
plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(original_image_pil)

plt.subplot(1, 3, 2)
plt.title('Transformed Image')
plt.imshow(transformed_image_pil)

plt.subplot(1, 3, 3)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_image)

plt.show()


# Convert the PIL Images to YCbCr color space
original_image_ycbcr = original_image_pil.convert('YCbCr')
transformed_image_ycbcr = transformed_image_pil.convert('YCbCr')
reconstructed_image_ycbcr = reconstructed_image.convert('YCbCr')

# Split the images into Y, Cb, and Cr channels
original_y, _, _ = original_image_ycbcr.split()
transformed_y, _, _ = transformed_image_ycbcr.split()
reconstructed_y, _, _ = reconstructed_image_ycbcr.split()

# Display the Y channel of the original, transformed, and reconstructed images side by side
plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image - Y Channel')
plt.imshow(original_y, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Transformed Image - Y Channel')
plt.imshow(transformed_y, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Reconstructed Image - Y Channel')
plt.imshow(reconstructed_y, cmap='gray')

plt.show()

plt.figure(figsize=(50, 20))

# Display the original image with BRISQUE score
plt.subplot(1, 2, 1)
plt.title(f'Ground Truth (BRISQUE: {brisque.score(original_image_pil):.2f})')
plt.imshow(original_image_np)

# Display the transformed image with BRISQUE score
plt.subplot(1, 2, 2)
plt.title(f'Bicubic Interpolation (BRISQUE: {brisque.score(transformed_image_pil):.2f})')
plt.imshow(transformed_image_np)


# Calculate and display the BRISQUE scores for the images
original_brisque_score = brisque.score(original_image_pil)
transformed_brisque_score = brisque.score(transformed_image_pil)
reconstructed_image_brisque_score = brisque.score(reconstructed_image)
print(f'Original Image BRISQUE Score: {original_brisque_score:.2f}')
print(f'Transformed Image BRISQUE Score: {transformed_brisque_score:.2f}')
print(f'Reconstructed Image BRISQUE Score: {reconstructed_image_brisque_score:.2f}')


# Print the shapes of the original, transformed, and reconstructed images
print(f'Original Image Shape: {original_image.size()}')
print(f'Transformed Image Shape: {transformed_image.size()}')
print(f'Reconstructed Image Shape: {reconstructed_image.size}')



# Define a custom PSNR calculation function
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100  # PSNR is infinity when the images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Calculate PSNR between the original and reconstructed images
psnr_reconstructed = calculate_psnr(np.array(original_image_pil), np.array(reconstructed_image))
print(f'PSNR between Original and Reconstructed Image: {psnr_reconstructed:.2f} dB')

# Calculate PSNR between the original and transformed images
psnr_transformed = calculate_psnr(np.array(original_image_pil), np.array(transformed_image_pil))
print(f'PSNR between Original and Transformed Image: {psnr_transformed:.2f} dB')

# Calculate SSIM between the original and reconstructed images
ssim_reconstructed = compare_ssim(np.array(original_image_ycbcr), np.array(reconstructed_image_ycbcr), data_range=255, channel_axis=2)
print(f'SSIM between Original and Reconstructed Image: {ssim_reconstructed:.2f}')

# Calculate SSIM between the original and transformed images
ssim_transformed = compare_ssim(np.array(original_image_ycbcr), np.array(transformed_image_ycbcr), data_range=255, channel_axis=2)
print(f'SSIM between Original and Transformed Image: {ssim_transformed:.2f}')

# Display the original, transformed, and reconstructed images side by side
plt.figure(figsize=(30, 10))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(original_image_pil)

plt.subplot(1, 3, 2)
plt.title(f'Transformed Image - Y Channel\n SSIM: {ssim_transformed:.2f} num_epochs: {num_epochs}' )
plt.imshow(transformed_image_pil)

plt.subplot(1, 3, 3)
plt.title(f'Reconstructed Image - Y Channel\n SSIM: {ssim_reconstructed:.2f} num_epochs: {num_epochs}')
plt.imshow(reconstructed_image)

plt.show()
