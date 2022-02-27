from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.make_dataset import ImageDataset
from src.models.model import Net

import torch
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn as nn

import argparse
import yaml

def set_global_args(args):
    config_file = args.config
    with open(config_file) as infile:
        config_dict = yaml.load(infile, Loader=yaml.SafeLoader)
    global DATA, TRAINING_HP, WANDB_KEY, PROJECT, VALIDATION

    DATA = config_dict['data']
    TRAINING_HP = config_dict['training']
    #WANDB_KEY = config_dict['wandb']
    PROJECT = config_dict['project']
    VALIDATION = config_dict['validation']


def run():
    parser = argparse.ArgumentParser(description='ML Pipe NN.')
    parser.add_argument("--config", help="Provide path to configuration file")
    args = parser.parse_args()

    set_global_args(args)

    transform = transforms.Resize((DATA['img-res'][0], DATA['img-res'][1]))

    dataset = ImageDataset(
                    data_dir=DATA['location'],
                    transform=transform)

    test_size = int(VALIDATION['test_size'] * len(dataset))
    train_size = len(dataset) - test_size
    trainset, testset = torch.utils.data.random_split(dataset, 
                    [train_size, test_size])

    trainloader = DataLoader(trainset, batch_size=TRAINING_HP['batch_size'], shuffle=True)
    testloader = DataLoader(testset, batch_size=TRAINING_HP['batch_size'])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not DATA['greyscale']:
        in_channels = 3
    else:
        in_channels = 1

    net = Net(in_channels=in_channels, num_classes=dataset.get_num_classes()).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=TRAINING_HP['learning_rate'])

    for epoch in range(TRAINING_HP['epochs']):
        for i, (images,labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = net(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{TRAINING_HP['epochs']}], " +
                    f"Step [{i+1}/{len(trainloader)}], Loss: {loss.item()}")

    #@title Evaluating the accuracy of the model

    correct = 0
    total = 0
    for images, labels in testloader:
        
        output = net(images)
        _, predicted = torch.max(output,1)
        correct += (predicted == labels).sum()
        total += labels.size(0)

    print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))


if __name__ == "__main__":
    run()
    
    
