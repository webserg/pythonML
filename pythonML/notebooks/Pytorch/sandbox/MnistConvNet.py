# The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of
# 10,000 examples. It is a subset of a larger set available from NIST.
# The digits have been size-normalized and centered in a fixed-size image.
# The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm.
# the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating
# the image so as to position this point at the center of the 28x28 field.

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


class MnistConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchNorm2(self.conv2(x))))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

    def train_net(self, train_loader, criterion, optimizer):
        print(self)
        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 2
    num_classes = 10
    batch_size = 100
    learning_rate = 0.001

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='~/.pytorch/MNIST_data/',
                                               train=True,
                                               transform=transform,
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='~/.pytorch/MNIST_data/',
                                              train=False,
                                              transform=transform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = MnistConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train_net(train_loader, criterion, optimizer)

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'models/mnistModel.ckpt')
