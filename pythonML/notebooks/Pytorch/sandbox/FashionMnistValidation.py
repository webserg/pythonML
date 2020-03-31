from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
import torch.nn as nn
from pythonML.notebooks.Pytorch.sandbox.FashionMnistConvNet import FashionMnistConvNet

if __name__ == '__main__':
    num_workers = 0
    batch_size = 20
    valid_size = 0.2

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    # specify the image classes
    classes = ['T-shirt/top',
               ' Trouser',
               ' Pullover',
               ' Dress',
               ' Coat',
               ' Sandal',
               ' Shirt',
               ' Sneaker',
               ' Bag',
               ' Ankle boot', ]

    model = FashionMnistConvNet()
    print(model)
    model.cuda()

    model.load_state_dict(torch.load('models/model_fashionMnist.pt'))
    # track test loss
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    criterion = nn.NLLLoss()
    model.eval()
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct_tensor = pred.eq(target.data.view_as(pred))
        correct = np.squeeze(correct_tensor.cpu().numpy())
        for i in range(batch_size):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    test_loss = test_loss / len(test_loader.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
