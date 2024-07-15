import os
import random
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ToPILImage, ToTensor
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import SRCNN, SRCNN2, SRCNN3
import helper
from helper import YChannelDataset, CbCrChannelDataset, PairedDataset

if __name__ == '__main__':
    # ------------------------- 0.SETUP ---------------------------------------------------------------- #
    # Define the directories for the training and evaluation datasets and get folder item extensions

    train_dir = 'your_path_to_DIV2K_train_HR'
    eval_dir = 'your_path_to_DIV2K_valid_HR'
    first_image = 'your_path_to_DIV2K_train_HR\\0001.png'
    img1_name, img1_extension = os.path.splitext(first_image)

    # ---------------------------------------------------------------------------------------------------------- #
    # ---------------- 1.SETUP THE DATA -------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------- #
    min_width, min_height = helper.find_min_dimensions(train_dir, img1_extension)

    # ---------------------------------------------------------------------------------------------------------- #
    # -------------------------------- 2. PREPARING THE DATA ------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------- #
    # Set downscale factor
    downscale_factor = 2
    # Define the batch size
    batch_size = 8
    # Create and prepare datasets
    # Train and eval datasets contain the original images from the DIV2K dataset
    # Bicubic dataset contains the downscale and then bicubic interpolated images
    train_dataset, eval_dataset, bicubic_dataset = helper.prepare_data(min_height, min_width, downscale_factor,
                                                                       train_dir, eval_dir)

    # Create new datasets that convert RGB images to YCbCr and split into Y, Cb, Cr channels
    # Notice that the original paper proposes an SRCNN model that only operates on the Y channel
    y_train_dataset = YChannelDataset(train_dataset)
    y_bicubic_dataset = YChannelDataset(bicubic_dataset)
    y_val_dataset = YChannelDataset(eval_dataset)
    cb_channel_dataset = CbCrChannelDataset(bicubic_dataset, 1)
    cr_channel_dataset = CbCrChannelDataset(bicubic_dataset, 2)

    # Create new data loaders
    # y_train_loader = DataLoader(y_train_dataset, batch_size=batch_size)
    # y_bicubic_loader = DataLoader(y_bicubic_dataset, batch_size=batch_size)
    # y_val_loader = DataLoader(y_val_dataset, batch_size=batch_size)

    # Create data loaders for the training, evaluation and bicubic datasets
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # bicubic_loader = torch.utils.data.DataLoader(bicubic_dataset, batch_size=batch_size, shuffle=False)
    # eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    # helper.print_data(train_dataset, bicubic_dataset, eval_dataset, train_loader, eval_loader, bicubic_loader,
    #                 batch_size)

    # ---------------------------------------------------------------------------------------------------------- #
    # ---------------- 3. VISUALIZING THE DATA -------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------- #

    # Get a random image from the datasets
    rand_index = random.randrange(0, len(train_dataset) - 1)
    original_image, original_label = train_dataset[rand_index]
    transformed_image, transformed_label = bicubic_dataset[rand_index]

    # Get the first image from each dataset
    y_train_image = y_train_dataset[rand_index]
    y_bicubic_image = y_bicubic_dataset[rand_index]

    # Create a new figure
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot the first image of y_train_dataset
    axs[0].imshow(y_train_image.squeeze(), cmap='gray')
    axs[0].set_title('Y Train Image')

    # Plot the first image of y_bicubic_dataset
    axs[1].imshow(y_bicubic_image.squeeze(), cmap='gray')
    axs[1].set_title('Y Bicubic Image')

    # Display the plot
    plt.show()

    # Define the data augmentation transforms
    hflip1 = RandomHorizontalFlip(p=1.0)  # p=1.0 to always apply the flip
    vflip1 = RandomVerticalFlip(p=1.0)

    # Convert tensors to PIL Images
    to_pil_image = ToPILImage()
    y_train_image_pil = to_pil_image(y_train_image.squeeze())
    y_bicubic_image_pil = to_pil_image(y_bicubic_image.squeeze())

    # Apply the transforms
    y_train_image_pil = hflip1(vflip1(y_train_image_pil))
    y_bicubic_image_pil = hflip1(vflip1(y_bicubic_image_pil))

    # Convert PIL Images back to tensors
    to_tensor = ToTensor()
    y_train_image = to_tensor(y_train_image_pil)
    y_bicubic_image = to_tensor(y_bicubic_image_pil)

    # Create a new figure
    fig, axs = plt.subplots(1, 2, figsize=(40, 20))
    # Plot the first image of y_train_dataset
    axs[0].imshow(y_train_image.squeeze(), cmap='gray')
    axs[0].set_title('Y Train Image')

    # Plot the first image of y_bicubic_dataset
    axs[1].imshow(y_bicubic_image.squeeze(), cmap='gray')
    axs[1].set_title('Y Bicubic Image')

    # Display the plot
    plt.show()

    # ---------------------------------------------------------------------------------------------------------- #
    # ---------------- 4. TRAINING THE MODEL -------------------------------------------------------------- #
    # ---------------------------------------------------------------------------------------------------------- #

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print('Pytorch version:', torch.__version__)
    print('Cuda version:', torch.version.cuda)
    # Initialize the model
    model = SRCNN()
    model = model.to(device)  # Move the model to the GPU if available

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Define a learning rate scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=29)

    # Number of epochs
    num_epochs = 800

    # Define early stopping parameters
    patience = 60 # Number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')  # Initialize best_val_loss to infinity
    epochs_without_improvement = 0  # Number of epochs without improvement

    # Specify the save directory
    directory = 'your_path_to_save_directory'

    # Initialize a list to hold the average loss per epoch
    avg_losses = []
    val_losses = []

    # Create a data loader for your training dataset
    paired_train_dataset = PairedDataset(y_train_dataset, y_bicubic_dataset)
    y_train_loader = DataLoader(paired_train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
                                pin_memory=True)
    y_val_loader = DataLoader(y_val_dataset, batch_size=batch_size)
    cb_channel_loader = DataLoader(cb_channel_dataset, batch_size=batch_size)
    cr_channel_loader = DataLoader(cr_channel_dataset, batch_size=batch_size)

    helper.print_data(train_dataset, bicubic_dataset, eval_dataset, y_train_loader, y_val_loader,
                      batch_size)

    flag=0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (y_train_images, y_bicubic_images) in enumerate(tqdm(y_train_loader)):
            optimizer.zero_grad()  # Reset gradients tensors

            # Apply the same transformation to both images with a probability of 1/3
            if random.random() < 0.11:
                # Horizontal flip
                y_train_images = torch.flip(y_train_images, [2])
                y_bicubic_images = torch.flip(y_bicubic_images, [2])
            elif random.random() < 0.33:
                # 90 degree rotation
                y_train_images = torch.rot90(y_train_images, 1, [2, 3])
                y_bicubic_images = torch.rot90(y_bicubic_images, 1, [2, 3])
            # else: do nothing

            y_train_images = y_train_images.to(device)
            y_bicubic_images = y_bicubic_images.to(device)

            # Forward pass
            outputs = model(y_bicubic_images)

            # Calculate loss
            loss = criterion(outputs, y_train_images)

            # Backward pass
            loss.backward()

            running_loss += loss.item()

            optimizer.step()  # Now we can do an optimizer step

        avg_loss = running_loss / len(y_train_loader)
        avg_losses = np.append(avg_losses, avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss}')

        # Validation
        model.eval()  # Set the model to evaluation mode
        with torch.inference_mode():  # Temporarily turn off gradient descent
            val_loss = 0.0

            for i, y_val_images in enumerate(y_val_loader):  # Loop over batches in the validation dataset
                y_val_images = y_val_images.to(device)  # Move images to the device (CPU or GPU)
                outputs = model(y_val_images)  # Forward pass through the model
                loss = criterion(outputs, y_val_images)  # Calculate loss
                val_loss += loss.item()  # Accumulate loss

            val_loss /= len(y_val_loader)  # Calculate average loss over all batches
            val_losses.append(val_loss)
            print(f'Validation Loss: {val_loss}')
        # Update the learning rate
        # scheduler.step(val_loss)

        # Check for early stopping and save the best model so far
        if val_loss < best_val_loss:
            print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}.')
            best_val_loss = val_loss
            best_epoch = epoch + 1
        #    epochs_without_improvement = 0
        print('Saving model...')
        torch.save(model.state_dict(), os.path.join(directory, f'model_{epoch+1}.pth'))

      #  else:
       #     epochs_without_improvement += 1

        #if epochs_without_improvement == patience:
         #   print('Early stopping')
          #  break

    # Plot average loss per epoch
    # Plot average loss and validation loss per epoch
    plt.figure(figsize=(20, 10))
    plt.plot(range(1, len(avg_losses) + 1), avg_losses, label='Average Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'Best epoch: {best_epoch}')
