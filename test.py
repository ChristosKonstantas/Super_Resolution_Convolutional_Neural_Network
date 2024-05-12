import cv2
import numpy as np
import torch
from model import SRCNN, SRCNN2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import helper
from skimage.metrics import structural_similarity as compare_ssim

# For lr = 1e-5 best model is 74
# Load the trained model
# barbara.png , model_799, (5,5) blur kernel
# 35
model = SRCNN()
model.load_state_dict(torch.load('C:\\Work\\PycharmProjectsLaptop\\srcnn_final\\model_35.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

# Load your image
image = cv2.imread('C:\\Users\\Tito\\Desktop\\butterfly.png')
downscale_factor = 2
# Downscale the image by a factor of 4 and then upscale with bicubic interpolation
downscaled_image = cv2.resize(image, (image.shape[1]//downscale_factor, image.shape[0]//downscale_factor), interpolation=cv2.INTER_CUBIC)

upscaled_image = cv2.resize(downscaled_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

upscaled_image = cv2.GaussianBlur(upscaled_image, (5, 5), 0)


# Convert the input image to YCbCr
image_ycbcr = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2YCrCb)

# Extract the Y channel
y_channel = image_ycbcr[:,:,0]

# Convert to tensor and add batch dimension
input_tensor = torch.from_numpy(y_channel).unsqueeze(0).unsqueeze(0).float().to(device)

# Pass the input through the model
with torch.no_grad():
    output = model(input_tensor)

# Denormalize the output
super_resolved_y_channel = (output.squeeze().cpu().numpy()).astype(np.uint8)

# Perform bicubic interpolation on Cb and Cr channels
cb_channel = cv2.resize(image_ycbcr[:,:,1], dsize=(super_resolved_y_channel.shape[1], super_resolved_y_channel.shape[0]), interpolation=cv2.INTER_CUBIC)
cr_channel = cv2.resize(image_ycbcr[:,:,2], dsize=(super_resolved_y_channel.shape[1], super_resolved_y_channel.shape[0]), interpolation=cv2.INTER_CUBIC)

# Merge the super-resolved Y channel with bicubic interpolated Cb and Cr channels
super_resolved_ycbcr = np.stack((super_resolved_y_channel, cb_channel, cr_channel), axis=2)

# Convert back to RGB color space for viewing
super_resolved_image_rgb = cv2.cvtColor(super_resolved_ycbcr, cv2.COLOR_YCrCb2RGB)

# Calculate PSNR
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100  # PSNR is infinity when the images are identical
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Calculate PSNR between the original and bicubic upscaled images
psnr_bicubic_upscaled = calculate_psnr(image, upscaled_image)

# Calculate PSNR between the original and super-resolved images
psnr_super_resolved = calculate_psnr(image, super_resolved_image_rgb)

# Save the super-resolved image
cv2.imwrite('path_to_save_super_resolved_image.jpg', super_resolved_image_rgb)

# Display the original, bicubic upscaled, and super-resolved images with zoomed-in patches
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Define the region of interest in the upper right corner
roi = [250, 100, 70, 70]  # [xmin, ymin, width, height]

# Original image
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title(f'Original Image')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)
ax_zoom1 = fig.add_axes([0.3, 0.6, 0.1, 0.1])
ax_zoom1.imshow(cv2.cvtColor(image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

# Bicubic upscaled image
ax[1].imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
ax[1].set_title(f'Bicubic Upscaled Image\nPSNR: {psnr_bicubic_upscaled:.2f}')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[1].add_patch(rect)
ax_zoom2 = fig.add_axes([0.58, 0.6, 0.1, 0.1])  # Adjust these values
ax_zoom2.imshow(cv2.cvtColor(upscaled_image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

# Super-resolved image
ax[2].imshow(cv2.cvtColor(super_resolved_image_rgb, cv2.COLOR_BGR2RGB))
ax[2].set_title(f'Super-Resolved Image\nPSNR: {psnr_super_resolved:.2f}')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[2].add_patch(rect)
ax_zoom3 = fig.add_axes([0.85, 0.6, 0.1, 0.1])  # Adjust these values
ax_zoom3.imshow(cv2.cvtColor(super_resolved_image_rgb[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

plt.show()


# Calculate SSIM using scikit-image's compare_ssim function
ssim_bicubic_upscaled = compare_ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY))
ssim_super_resolved = compare_ssim(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(super_resolved_image_rgb, cv2.COLOR_BGR2GRAY))

# Save the super-resolved image
cv2.imwrite('path_to_save_super_resolved_image.jpg', super_resolved_image_rgb)

# Display the original, bicubic upscaled, and super-resolved images with SSIM values in the titles
fig, ax = plt.subplots(1, 3, figsize=(18, 6))


# Original image
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title(f'Original Image')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)
ax_zoom1 = fig.add_axes([0.3, 0.6, 0.1, 0.1])
ax_zoom1.imshow(cv2.cvtColor(image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

# Bicubic upscaled image
ax[1].imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
ax[1].set_title(f'Bicubic Upscaled Image\nSSIM: {ssim_bicubic_upscaled:.2f}')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[1].add_patch(rect)
ax_zoom2 = fig.add_axes([0.58, 0.6, 0.1, 0.1])  # Adjust these values
ax_zoom2.imshow(cv2.cvtColor(upscaled_image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

# Super-resolved image
ax[2].imshow(cv2.cvtColor(super_resolved_image_rgb, cv2.COLOR_BGR2RGB))
ax[2].set_title(f'Super-Resolved Image\nSSIM: {ssim_super_resolved:.2f}')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[2].add_patch(rect)
ax_zoom3 = fig.add_axes([0.85, 0.6, 0.1, 0.1])  # Adjust these values
ax_zoom3.imshow(cv2.cvtColor(super_resolved_image_rgb[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

plt.show()

# Calculate MSE
def calculate_mse(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    return mse

# Calculate MSE between the original and bicubic upscaled images
mse_bicubic_upscaled = calculate_mse(image, upscaled_image)

# Calculate MSE between the original and super-resolved images
mse_super_resolved = calculate_mse(image, super_resolved_image_rgb)

# Save the super-resolved image
cv2.imwrite('path_to_save_super_resolved_image.jpg', super_resolved_image_rgb)

# Display the original, bicubic upscaled, and super-resolved images with zoomed-in patches
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Original image
ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0].set_title(f'Original Image')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)
ax_zoom1 = fig.add_axes([0.3, 0.6, 0.1, 0.1])
ax_zoom1.imshow(cv2.cvtColor(image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

# Bicubic upscaled image
ax[1].imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
ax[1].set_title(f'Bicubic Upscaled Image\nMSE: {mse_bicubic_upscaled:.2f}')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[1].add_patch(rect)
ax_zoom2 = fig.add_axes([0.58, 0.6, 0.1, 0.1])
ax_zoom2.imshow(cv2.cvtColor(upscaled_image[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

# Super-resolved image
ax[2].imshow(cv2.cvtColor(super_resolved_image_rgb, cv2.COLOR_BGR2RGB))
ax[2].set_title(f'Super-Resolved Image\nMSE: {mse_super_resolved:.2f}')
rect = patches.Rectangle((roi[0], roi[1]), roi[2], roi[3], linewidth=1, edgecolor='r', facecolor='none')
ax[2].add_patch(rect)
ax_zoom3 = fig.add_axes([0.85, 0.6, 0.1, 0.1])
ax_zoom3.imshow(cv2.cvtColor(super_resolved_image_rgb[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]], cv2.COLOR_BGR2RGB))

plt.show()
