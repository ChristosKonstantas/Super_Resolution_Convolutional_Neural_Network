from torch import nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class SRCNN2(nn.Module):
    def __init__(self):
        super(SRCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)  # Additional hidden layer
        self.conv4 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)  # Output layer
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))  # Additional ReLU activation function
        x = self.conv4(x)
        return x

# SRCNN model with transposed convolution layer for upscaling
class SRCNN3(nn.Module):
    def __init__(self, scale_factor=2):
        super(SRCNN3, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.upscale = nn.ConvTranspose2d(1, 1, kernel_size=4*scale_factor, stride=scale_factor, padding=scale_factor - 1, output_padding=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.upscale(x)  # Upscale using transposed convolution
        return x

# SRCNN model with transposed convolution layer for upscaling
class SRCNN4(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x