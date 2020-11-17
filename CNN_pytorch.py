import torch
import torchvision
import torch.nn as nn
from torch.optim import SGD, Adam
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms

"""
Implementation of a CUSTOM CNN (Convolutional Neural Network) to predict image attributes
Implemented CNN demonstrates importance of choosing RIGHT kernel size and BatchNorm effects on model accuracy
Also Visualizes Feature maps from layers and Filters of Convolutional Layers
"""

class Net(nn.Module):
    """
    Base Model holder class
    """
    def __init__(self):
        """
        Initialization, nothing fancy here
        """
        super(Net, self).__init__()
        self.conv_layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11),
                                         nn.ReLU(inplace=True), nn.BatchNorm2d(64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                                         nn.ReLU(inplace=True), nn.BatchNorm2d(128))
        self.conv_layer3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
                                         nn.ReLU(inplace=True), nn.BatchNorm2d(128))
        self.pool2 = nn.AvgPool2d(kernel_size=4)

        self.fc1 = nn.Sequential(nn.Linear(in_features=128, out_features=10))

    def forward(self, x):
        """
        Forward pass, pay attention to dimensions
        """
        x = self.pool1(self.conv_layer1(x))
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


def main():
    """
    Driver function
    """
    device = torch.device("cuda")
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=2000,
                                               shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=2000,
                                              shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # init model
    model = Net()
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    optimizer = Adam(model.parameters())
    model.to("cuda:0")
    history = []
    epochs = 10
    # Standard Torch Training loop
    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0

        valid_loss = 0.0
        valid_acc = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)
            print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

        with torch.no_grad():

            model.eval()

            for j, (inputs, labels) in enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = loss_criterion(outputs, labels)

                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                valid_acc += acc.item() * inputs.size(0)

                print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j,
                                                                                                           loss.item(),
                                                                                                           acc.item()))

        avg_train_loss = train_loss / len(trainset)
        avg_train_acc = train_acc / float(len(trainset))

        avg_valid_loss = valid_loss / len(testset)
        avg_valid_acc = valid_acc / float(len(testset))

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

        print(
            "Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%".format(
                epoch, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100))

    history = np.array(history)
    plt.plot(history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, 3)
    plt.savefig('loss_curve3.png')

    plt.plot(history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.savefig('accuracy_curve3.png')


    # Visualize what the CNN filters look like

    model = model.to("cpu")
    model_weights = []
    conv_layers = []
    conv_layers.append(list(model.conv_layer1.children())[0])
    model_weights.append(model.conv_layer1[0].weight)

    plt.figure(figsize=(4, 16))
    for i, filter in enumerate(model_weights[0]):
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter[0, :, :].detach())
        plt.axis('off')
        plt.savefig('filter3_0.png')

    img = cv.imread(f"D:\RIT\CV\Scripts\Assignment2\Abyssinian.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # define the transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    img = np.array(img)
    # apply the transforms
    img = transform(img)
    print(img.size())
    # unsqueeze to add a batch dimension
    img = img.unsqueeze(0)
    print(img.size())
    # pass the image through all the layers
    results = [conv_layers[0](img)]
    for i in range(1, len(conv_layers)):
        # pass the result from the last layer to the next layer
        results.append(conv_layers[i](results[-1]))
    # make a copy of the `results`
    outputs = results

    # visualize 64 features from each layer
    # (although there are more feature maps in the upper layers)
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(30, 30))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print(layer_viz.size())
        for i, filter in enumerate(layer_viz):
            if i == 64:  # we will visualize only 8x8 blocks from each layer
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(filter)
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"filter3_1.png")
        plt.show()


if __name__ == '__main__':
    main()
