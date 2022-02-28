import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from sklearn.model_selection import KFold

from src.data.make_dataset import ImageDataset
from src.models.model import Net


class MLPipe():
    def __init__(self, config_dict=None):
        self.DATA = config_dict['data']
        self.TRAINING_HP = config_dict['training']
        #self.WANDB_KEY = config_dict['wandb']
        self.PROJECT = config_dict['project']
        self.VALIDATION = config_dict['validation']

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def hold_out_split(self):
        test_size = int(self.VALIDATION['test_size'] * len(self.dataset))
        train_size = len(self.dataset) - test_size
        trainset, testset = torch.utils.data.random_split(self.dataset, 
                        [train_size, test_size])

        trainloader = DataLoader(trainset, 
                                batch_size=self.TRAINING_HP['batch_size'], 
                                shuffle=True)
        
        testloader = DataLoader(testset, 
                                batch_size=self.TRAINING_HP['batch_size'])

        return trainloader, testloader

    def k_fold_split(self):
        kfold = KFold(n_splits=self.VALIDATION['folds'], shuffle=True)
        trainloader, testloader = self.hold_out_split()
        trainloader = ConcatDataset([trainloader])
        trainloader_list, valloader_list = [], []
        for fold, (train_ids, test_ids) in enumerate(kfold.split(trainloader)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            
            trainloader_list.append(torch.utils.data.DataLoader(
                            trainloader, 
                            batch_size=self.TRAINING_HP['batch_size'],
                            sampler=train_subsampler))

            valloader_list.append(torch.utils.data.DataLoader(
                            trainloader,
                            batch_size=self.TRAINING_HP['batch_size'],
                            sampler=val_subsampler))

        return trainloader_list, valloader_list, testloader

    def preproc_data(self):
        transform = transforms.Resize((
                        self.DATA['img-res'][0], 
                        self.DATA['img-res'][1]))

        self.dataset = ImageDataset(
                        data_dir=self.DATA['location'],
                        transform=transform)

        if self.VALIDATION['folds']:
            self.trainloader, self.valloader, self.testloader = self.k_fold_split()
        else:
            self.trainloader, self.testloader = self.hold_out_split()
            

    def train(self):
        if not self.DATA['greyscale']:
            in_channels = 3
        else:
            in_channels = 1

        self.net = Net(in_channels=in_channels, num_classes=self.dataset.get_num_classes()).to(self.device)

        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.TRAINING_HP['learning_rate'])

        if self.VALIDATION['folds']:
            pass # TODO: implement cross validation training loop
        else:
            for epoch in range(self.TRAINING_HP['epochs']):
                for i, (images,labels) in enumerate(self.trainloader):
                    optimizer.zero_grad()
                    outputs = self.net(images)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    if (i+1) % 100 == 0:
                        print(f"Epoch [{epoch+1}/{self.TRAINING_HP['epochs']}], " +
                            f"Step [{i+1}/{len(self.trainloader)}], Loss: {loss.item()}")


    def eval(self):
        correct = 0
        total = 0
        for images, labels in self.testloader:
            
            output = self.net(images)
            _, predicted = torch.max(output,1)
            correct += (predicted == labels).sum()
            total += labels.size(0)

        print('Accuracy of the model: %.3f %%' %((100*correct)/(total+1)))