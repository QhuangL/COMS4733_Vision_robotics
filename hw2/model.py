import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniUNet(nn.Module):
    # TODO: implement a neural network as described in the handout
    def __init__(self):
        """Initialize the layers of the network as instance variables."""
        super(MiniUNet, self).__init__()
        # TODO

        self.relu = nn.ReLU(inplace=True)
        # left
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)

        # right
        self.conv6 = nn.Conv2d(128+256, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(64+128, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(32+64, 32, 3, padding=1)
        self.conv9 = nn.Conv2d(16+32, 16, 3, padding=1)
        self.conv10 = nn.Conv2d(16, 6, 1)

    def forward(self, x):
        """
        In:
            x: Tensor [batchsize, channel, height, width], channel=3 for rgb input
        Out:
            output: Tensor [batchsize, class, height, width], class=number of objects + 1 for background
        Purpose:
            Forward process. Pass the input x through the layers defined in __init__() to get the output.
        """
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(self.pool1(x1)))
        x3 = self.relu(self.conv3(self.pool1(x2)))
        x4 = self.relu(self.conv4(self.pool1(x3)))
        x5 = self.relu(self.conv5(self.pool1(x4)))

        x = self.relu(self.conv6(torch.cat([F.interpolate(x5, scale_factor=2), x4], 1)))
        x = self.relu(self.conv7(torch.cat([F.interpolate(x, scale_factor=2), x3], 1)))
        x = self.relu(self.conv8(torch.cat([F.interpolate(x, scale_factor=2), x2], 1)))
        x = self.relu(self.conv9(torch.cat([F.interpolate(x, scale_factor=2), x1], 1)))

        output = self.conv10(x)
        return output


if __name__ == '__main__':
    model = MiniUNet()
    input_tensor = torch.zeros([1, 3, 240, 320])
    output = model(input_tensor)
    print("output size:", output.size())
    print(model)
